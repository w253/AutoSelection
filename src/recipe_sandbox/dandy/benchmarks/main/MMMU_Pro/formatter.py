"""将MMMU-Pro vision/test转换为DataRecipe统一benchmark格式。

脚本读取全部Parquet分片，解析并校验候选项与答案，按原始ID稳定编号，过滤无法
用统一规则评分的答案后按subject抽样；随后复制图片、写出formatted_data/data.json
并执行字段、数量、ID和图片引用校验。text_input直接使用六项benchmark共享的
回答格式指令；运行参数与采样规模均通过文件常量控制。
"""

import ast
import hashlib
import json
import random
import re
import shutil
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

import pyarrow as arrow
import pyarrow.parquet as parquet

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from recipe_sandbox.dandy.benchmarks.sampling import balanced_eval_records

from recipe_sandbox.dandy.data.answer import (  # noqa: E402
    append_final_answer_instruction,
    partition_supported_answers,
)


# MMMU-Pro 提供三种 test 配置：
# - standard (4 options)：原始视觉材料与独立的题目文本、4个候选项共同输入；
# - standard (10 options)：仍为图片加独立文本，但扩充候选项以降低随机猜测收益；
# - vision：将题目、选项和原始视觉材料统一渲染到一张图片中，不再单独提供
#   题目文本。当前按既定方案选择 vision/test，它天然符合单个 img_path 的输出
#   结构，因此 text_input 不重复题面，只承载与SFT/GRPO训练数据一致的最终答案
#   格式指令（题目仍需模型从图片中读取）。每条20样本的对照实验显示，加入该指令
#   后本benchmark准确率由10%提升到30%；两个 standard 配置不参与本次转换。
#
# vision/test 共 1,730 条、30个 subject，各学科全量数量为：
# Accounting 58；Agriculture 60；Architecture_and_Engineering 60；Art 53；
# Art_Theory 55；Basic_Medical_Science 52；Biology 59；Chemistry 60；
# Clinical_Medicine 59；Computer_Science 60；Design 60；
# Diagnostics_and_Laboratory_Medicine 60；Economics 59；Electronics 60；
# Energy_and_Power 58；Finance 60；Geography 52；History 56；Literature 52；
# Manage 50；Marketing 59；Materials 60；Math 60；Mechanical_Engineering 59；
# Music 60；Pharmacy 57；Physics 60；Psychology 60；Public_Health 58；
# Sociology 54。
# 默认不设学科上限，使用200条eval总预算；SAE另算。
# options 虽以10项为主，但实际数量分布为2:28、3:42、4:125、5:41、6:16、
# 7:12、8:31、9:221、10:1213、12:1，因此代码按每条记录的真实选项数校验，
# 不假设全部样本恰好有10项。options 已绘制在 vision 图片内，不写入data.json。
BENCHMARK_NAME = "MMMU-Pro"
# 固定扩容前的顺序参考配额，仅用于保留原有50条SAE样本，不随eval规模变化。
SAE_REFERENCE_SAMPLE_COUNT = 7
VISION_DIR = Path("vision")
TEXT_INPUT = append_final_answer_instruction("")


def find_vision_parquets(vision_dir: Path) -> List[Path]:
    """定位 vision/test 的全部 Parquet 分片。

    功能与逻辑：
        在 vision/ 中按文件名固定顺序查找 test-*.parquet，要求至少存在一个
        分片，避免误读 standard 配置或其他文件。

    输入：
        vision_dir：MMMU-Pro 的 vision/ 原始数据目录。

    输出：
        按路径名称稳定排序的全部 test Parquet 分片。
    """
    if not vision_dir.is_dir():
        raise FileNotFoundError("找不到 vision 原始数据目录: {}".format(vision_dir))

    parquet_paths = sorted(vision_dir.glob("test-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("vision/ 中找不到 test-*.parquet")

    return parquet_paths


def build_original_id_sort_key(original_id: str) -> Tuple[str, int, str]:
    """为 MMMU-Pro 原始 ID 构造确定性排序键。

    功能与逻辑：
        将末尾形如 _数字 的部分按整数排序，前缀按不区分大小写的文本排序；若
        ID 不符合该形式，则以完整文本作为前缀，从而保证任何合法 ID 都能稳定
        排列。

    输入：
        original_id：原始样本 ID，例如 test_History_1。

    输出：
        可供 sorted 使用的三元组排序键。
    """
    matched_id = re.fullmatch(r"(.*)_(\d+)", original_id)

    if matched_id is None:
        return original_id.casefold(), -1, original_id

    return matched_id.group(1).casefold(), int(matched_id.group(2)), original_id


def parse_options(options_value: str, original_id: str) -> List[str]:
    """解析并校验字符串形式的候选选项。

    功能与逻辑：
        使用 ast.literal_eval 将 options 的字符串化列表安全解析为 Python 列表，
        再要求至少两个非空选项，避免假设每条样本固定为10个选项。

    输入：
        options_value：Parquet 中的 options 字符串。
        original_id：用于异常定位的原始样本 ID。

    输出：
        按原始顺序排列的候选选项字符串列表。
    """
    parsed_options = ast.literal_eval(options_value)

    if not isinstance(parsed_options, list):
        raise TypeError("样本 {} 的 options 解析后不是 list".format(original_id))
    if len(parsed_options) < 2:
        raise ValueError("样本 {} 的候选选项少于2个".format(original_id))

    normalized_options = [str(option).strip() for option in parsed_options]
    if any(not option for option in normalized_options):
        raise ValueError("样本 {} 存在空候选选项".format(original_id))

    return normalized_options


def load_and_number_records(parquet_paths: List[Path]) -> List[Dict[str, Any]]:
    """读取、校验并稳定编号全部 vision/test 样本。

    功能与逻辑：
        用 pyarrow 读取全部分片并合并，校验必需字段、ID 唯一性、subject、答案、
        选项和图片二进制；随后按原始 ID 稳定排序，在采样前为全部1,730条样本
        分配固定 id 与图片文件名。

    输入：
        parquet_paths：按固定顺序排列的 vision/test Parquet 分片。

    输出：
        已完成全量稳定编号和字段转换的样本列表，内部字段以 _ 开头。
    """
    tables = [parquet.read_table(parquet_path) for parquet_path in parquet_paths]
    table = arrow.concat_tables(tables)
    required_fields = {"id", "image", "options", "answer", "subject"}
    missing_fields = required_fields - set(table.column_names)

    if missing_fields:
        raise ValueError("vision/test 缺少字段: {}".format(sorted(missing_fields)))

    raw_records = table.to_pylist()
    if not raw_records:
        raise ValueError("vision/test 中没有样本")

    seen_original_ids: Set[str] = set()
    for row_number, record in enumerate(raw_records, start=1):
        original_id = str(record["id"]).strip()

        if not original_id:
            raise ValueError("第 {} 条样本的 id 为空".format(row_number))
        if original_id in seen_original_ids:
            raise ValueError("发现重复原始 ID: {}".format(original_id))
        if not str(record["subject"]).strip():
            raise ValueError("样本 {} 的 subject 为空".format(original_id))
        if not isinstance(record["image"], dict):
            raise TypeError("样本 {} 的 image 必须是 dict".format(original_id))
        if not isinstance(record["image"].get("bytes"), bytes):
            raise TypeError("样本 {} 缺少图片二进制数据".format(original_id))
        if not record["image"]["bytes"]:
            raise ValueError("样本 {} 的图片二进制为空".format(original_id))

        image_path = str(record["image"].get("path", "")).strip()
        image_suffix = Path(image_path).suffix.lower()
        if not image_suffix:
            raise ValueError("样本 {} 的图片路径没有扩展名".format(original_id))

        options = parse_options(str(record["options"]), original_id)
        answer = str(record["answer"]).strip().upper()
        valid_answers = set(string.ascii_uppercase[:len(options)])
        if answer not in valid_answers:
            raise ValueError(
                "样本 {} 的答案 {} 超出 {} 个候选项范围".format(
                    original_id,
                    answer,
                    len(options),
                )
            )

        seen_original_ids.add(original_id)

    sorted_records = sorted(
        raw_records,
        key=lambda record: build_original_id_sort_key(str(record["id"]).strip()),
    )
    numbered_records: List[Dict[str, Any]] = []

    for sample_number, record in enumerate(sorted_records, start=1):
        original_id = str(record["id"]).strip()
        image_path = str(record["image"]["path"]).strip()
        image_suffix = Path(image_path).suffix.lower()
        image_name = "{}-img-{:04d}{}".format(
            BENCHMARK_NAME,
            sample_number,
            image_suffix,
        )
        options = parse_options(str(record["options"]), original_id)

        numbered_records.append(
            {
                "id": "{}-{:04d}".format(BENCHMARK_NAME, sample_number),
                "img_path": "img/{}".format(image_name),
                "text_input": TEXT_INPUT,
                "answer": str(record["answer"]).strip().upper(),
                "original_id": original_id,
                "subject": str(record["subject"]).strip(),
                "_global_order": sample_number,
                "_option_count": len(options),
                "_image_name": image_name,
                "_image_bytes": record["image"]["bytes"],
            }
        )

    return numbered_records


def sample_records_by_subject(
    records: List[Dict[str, Any]],
    samples_per_subject: Optional[int],
    sampling_mode: Literal["sequential", "random"],
) -> List[Dict[str, Any]]:
    """按每个 subject 相同上限执行采样。

    功能与逻辑：
        在全量稳定编号后按 subject 分组。每个学科最多保留指定数量；顺序模式取
        组内全量编号最靠前的记录，随机模式无固定种子随机抽取。容量不足时保留
        该学科全部记录且不跨学科补配额，最终恢复为全量编号顺序。

    输入：
        records：已经完成全量稳定编号的 vision/test 样本。
        samples_per_subject：每个学科最多采样数；None表示保留全部输入记录。
        sampling_mode：仅允许 "sequential" 或 "random"。

    输出：
        按全量编号顺序排列的学科均衡采样结果。
    """
    if sampling_mode not in {"sequential", "random"}:
        raise ValueError("sampling_mode 只能是 'sequential' 或 'random'")
    if samples_per_subject is None:
        return list(records)
    if samples_per_subject <= 0:
        raise ValueError("samples_per_subject 必须为正整数或 None")

    subject_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        subject_groups[record["subject"]].append(record)

    sampled_records: List[Dict[str, Any]] = []
    for subject in sorted(subject_groups):
        subject_records = subject_groups[subject]
        sample_size = min(samples_per_subject, len(subject_records))

        if sampling_mode == "sequential":
            selected_records = subject_records[:sample_size]
        else:
            selected_records = random.sample(subject_records, k=sample_size)

        sampled_records.extend(selected_records)

    return sorted(sampled_records, key=lambda record: record["_global_order"])


def print_distribution(records: List[Dict[str, Any]], title: str) -> None:
    """打印 subject、选项数量和答案分布。

    功能与逻辑：
        统计样本的学科、真实候选项数量和正确答案字母，用于核对全量数据与采样
        结果是否符合预期。

    输入：
        records：需要统计的样本列表。
        title：当前统计块标题。

    输出：
        无返回值；将分布直接打印到终端。
    """
    subject_counter = Counter(record["subject"] for record in records)
    option_count_counter = Counter(record["_option_count"] for record in records)
    answer_counter = Counter(record["answer"] for record in records)

    print("\n{} {} {}".format("=" * 20, title, "=" * 20))
    print("样本数: {}".format(len(records)))
    print("subject 分布:")
    for subject, count in sorted(subject_counter.items()):
        print("  {}: {}".format(subject, count))

    print("选项数量分布: {}".format(dict(sorted(option_count_counter.items()))))
    print("答案分布: {}".format(dict(sorted(answer_counter.items()))))


def prepare_output_dir(output_dir: str) -> Path:
    """安全清除旧转换结果并创建新的 img/ 目录。

    功能与逻辑：
        限制输出目录必须位于当前 MMMU-Pro 根目录内部，并拒绝当前目录、vision/
        及其子目录。若输出目录已存在，则完整删除后重新创建 img/。

    输入：
        output_dir：转换结果目录，默认为 "formatted_data"。

    输出：
        新创建的 img/ 目录 Path。
    """
    output_path = Path(output_dir).resolve()
    current_path = Path(".").resolve()
    vision_path = VISION_DIR.resolve()

    if output_path == current_path:
        raise ValueError("output_dir 不能是当前 MMMU-Pro 根目录")
    if current_path not in output_path.parents:
        raise ValueError("output_dir 必须位于当前 MMMU-Pro 目录内部")
    if output_path == vision_path or vision_path in output_path.parents:
        raise ValueError("output_dir 不能是 vision/ 或位于原始 vision/ 内部")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("output_dir 已存在但不是目录: {}".format(output_path))

        print("清除上一次转换结果: {}".format(output_path))
        shutil.rmtree(output_path)

    image_dir = output_path / "img"
    image_dir.mkdir(parents=True)
    return image_dir


def write_images(records: List[Dict[str, Any]], image_dir: Path) -> None:
    """将采样记录的 vision 图片写入 img/。

    功能与逻辑：
        将 Parquet 的 image.bytes 原样写入全量编号阶段确定的文件名；每条记录
        对应一张已包含题目、选项和视觉材料的合成图片。

    输入：
        records：采样后的样本列表。
        image_dir：最终图片输出目录。

    输出：
        无返回值；在 image_dir 中生成图片文件。
    """
    for record in records:
        (image_dir / record["_image_name"]).write_bytes(record["_image_bytes"])


def select_sae_records(
    eligible_records: List[Dict[str, Any]],
    reference_eval_records: List[Dict[str, Any]],
    count: int = 50,
) -> List[Dict[str, Any]]:
    """从原始小规模顺序评测集之外按稳定ID哈希固定划出SAE样本。

    输入：合格记录、原有顺序评测记录及专用样本数。
    输出：恢复全量编号顺序的SAE记录；可用补充样本不足时抛错。
    """
    reference_ids = {record["id"] for record in reference_eval_records}
    remaining = [record for record in eligible_records if record["id"] not in reference_ids]
    if len(remaining) < count:
        raise ValueError("原始参考评测集之外不足{}条SAE样本".format(count))
    selected = sorted(
        remaining,
        key=lambda record: (hashlib.sha256(record["id"].encode("utf-8")).digest(), record["id"]),
    )[:count]
    return sorted(selected, key=lambda record: record["_global_order"])


def write_data_json(
    records: List[Dict[str, Any]],
    sae_records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """写入统一格式的 data.json。

    功能与逻辑：
        移除所有以下划线开头的运行时字段，仅保留评测和回溯所需字段，并写入
        创建时间、样本数量和样本数组。

    输入：
        records：采样并按全量编号排序后的评测样本列表。
        sae_records：与评测ID互斥的SAE专用样本列表。
        output_dir：包含 data.json 和 img/ 的输出目录。

    输出：
        无返回值；在 output_dir 中生成 data.json。
    """
    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    public_sae_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in sae_records
    ]
    output_data = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "num_samples": len(public_records),
        "data": public_records,
        "num_sae_samples": len(public_sae_records),
        "sae_data": public_sae_records,
    }
    (output_dir / "data.json").write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_output(
    records: List[Dict[str, Any]],
    sae_records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """校验 data.json 与 img/ 的完整性。

    功能与逻辑：
        重新读取 data.json，核对样本数量、必需字段、唯一 ID、空 text_input、
        内部字段清理和所有图片路径，同时要求 img/ 文件数与样本数一致。

    输入：
        records：写入前的评测样本。
        sae_records：写入前的SAE专用样本。
        output_dir：转换结果目录。

    输出：
        无返回值；校验失败时抛出异常，成功时打印摘要。
    """
    saved_data = json.loads((output_dir / "data.json").read_text(encoding="utf-8"))
    saved_records = saved_data["data"]
    saved_sae_records = saved_data["sae_data"]

    if saved_data["num_samples"] != len(records):
        raise ValueError("data.json 的 num_samples 与实际样本数不一致")
    if len(saved_records) != len(records):
        raise ValueError("data.json 的 data 长度与实际样本数不一致")
    if saved_data["num_sae_samples"] != len(sae_records) or len(saved_sae_records) != len(sae_records):
        raise ValueError("data.json 的SAE样本数量不一致")

    required_fields = {"id", "img_path", "text_input", "answer"}
    saved_ids: Set[str] = set()
    for record in saved_records + saved_sae_records:
        missing_fields = required_fields - set(record)
        if missing_fields:
            raise ValueError("输出样本缺少字段: {}".format(sorted(missing_fields)))
        if record["id"] in saved_ids:
            raise ValueError("输出中存在重复 id: {}".format(record["id"]))
        if record["text_input"] != TEXT_INPUT:
            raise ValueError("vision 样本的 text_input 必须与统一格式指令一致")
        if any(key.startswith("_") for key in record):
            raise ValueError("data.json 中仍包含内部字段")
        if not (output_dir / record["img_path"]).is_file():
            raise FileNotFoundError("输出图片不存在: {}".format(record["img_path"]))

        saved_ids.add(record["id"])

    output_images = [path for path in (output_dir / "img").iterdir() if path.is_file()]
    if len(output_images) != len(records) + len(sae_records):
        raise ValueError("img/ 文件数与样本数不一致")

    print("\n转换结果校验通过")
    print("样本数: {}".format(len(saved_records)))
    print("SAE专用样本数: {}".format(len(saved_sae_records)))
    print("图片数: {}".format(len(output_images)))
    print("输出目录: {}".format(output_dir))


def convert_dataset(
    output_dir: str = "formatted_data",
    samples_per_subject: Optional[int] = None,
    sampling_mode: Literal["sequential", "random"] = "sequential",
    print_distribution_info: bool = True,
    target_sample_count: Optional[int] = 200,
) -> None:
    """完成 MMMU-Pro/vision/test 的统一格式转换。

    功能与逻辑：
        先安全清除旧输出，读取并稳定编号全部1,730条vision/test记录；使用统一
        答案规则排除无法评分的记录，先固定划出50条SAE专用样本，再从余下记录
        按每个subject相同上限采样评测题，写出两组图片和data.json并校验互斥ID。

    输入：
        target_sample_count：排除SAE后的eval总数，默认200；None取消总量限制。
        output_dir：输出目录，默认为 "formatted_data"。
        samples_per_subject：每个学科最多采样数，默认None，不限制每组数量；None表示全部非SAE合格题。
        sampling_mode：仅允许 "sequential" 或 "random"，默认顺序采样。
        print_distribution_info：是否打印全量和采样后的分布。

    输出：
        无返回值；在 output_dir 中生成 data.json 和 img/。
    """
    image_dir = prepare_output_dir(output_dir)
    parquet_paths = find_vision_parquets(VISION_DIR)
    numbered_records = load_and_number_records(parquet_paths)
    all_records, rejected_records = partition_supported_answers(numbered_records)
    print("排除无法规则评分的记录: {}".format(len(rejected_records)))
    reference_eval_records = sample_records_by_subject(
        all_records, SAE_REFERENCE_SAMPLE_COUNT, "sequential"
    )
    sae_records = select_sae_records(all_records, reference_eval_records)
    sae_ids = {record["id"] for record in sae_records}
    eval_candidates = [record for record in all_records if record["id"] not in sae_ids]
    sampled_records = sample_records_by_subject(
        eval_candidates,
        samples_per_subject,
        sampling_mode,
    )
    sampled_records = balanced_eval_records(
        sampled_records, target_sample_count, (('subject',),), sampling_mode
    )

    if print_distribution_info:
        print_distribution(all_records, "全量 vision/test")
        print_distribution(sampled_records, "采样后数据")

    write_images(sampled_records + sae_records, image_dir)
    write_data_json(sampled_records, sae_records, image_dir.parent)
    validate_output(sampled_records, sae_records, image_dir.parent)


if __name__ == "__main__":
    # 转换结果目录；重复运行时先清除此目录，不会修改三个官方配置目录。
    OUTPUT_DIR = "formatted_data"

    # 不限制组容量；下方总量200按原分层均衡分配。
    SAMPLES_PER_SUBJECT = None
    TARGET_SAMPLE_COUNT = 200

    # 可选值："sequential" 或 "random"；默认优先顺序，随机模式不设置种子。
    SAMPLING_MODE = "sequential"

    # True：打印全量和采样后分布；False：不打印。
    PRINT_DISTRIBUTION_INFO = True

    convert_dataset(
        output_dir=OUTPUT_DIR,
        samples_per_subject=SAMPLES_PER_SUBJECT,
        sampling_mode=SAMPLING_MODE,
        print_distribution_info=PRINT_DISTRIBUTION_INFO,
        target_sample_count=TARGET_SAMPLE_COUNT,
    )
