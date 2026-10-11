"""将MathVista testmini转换为DataRecipe统一benchmark格式。

脚本读取原始Parquet，稳定编号后筛选英文多项选择题，将答案文本唯一映射为选项
字母并过滤映射失败的记录，再按task与context均衡抽样。随后复制图片、写出
formatted_data/data.json；官方query后用独立分隔区块追加共享回答格式指令，
最后核对筛选条件、字段、分布、ID和图片引用。
"""

import hashlib
import json
import random
import shutil
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

import pyarrow.parquet as parquet

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from recipe_sandbox.dandy.data.answer import (  # noqa: E402
    append_final_answer_instruction,
    classify_answer,
)


# MathVista/testmini 共 1,000 条，其中当前只保留同时满足以下条件的数据：
# - question_type=multi_choice：多项选择题；
# - metadata.language=english：英文题目。
# 筛选后共有 478 条，answer_type 均为 text，且 choices 均非空。
# metadata.category 中 general-vqa 与 math-targeted-vqa 各 239 条。
#
# metadata.task 表示 5 类数学视觉任务，筛选后的数量为：
# - figure question answering（图形与图表问答）：116；
# - geometry problem solving（几何问题求解）：141；
# - math word problem（数学文字题）：22；
# - textbook question answering（教材问题问答）：107；
# - visual question answering（通用视觉问答）：92。
# metadata.context 表示图像的视觉形式，共涉及 16 类，例如几何图、函数图、
# 柱状图、科学图、自然图像和合成场景等；筛选后实际存在 20 个 task/context
# 组合。当前采样先在 task 间容量感知地均衡分配总配额，再在每个 task 内的
# context 间均衡分配。question_type、language 和 answer_type 筛选后均为常量，
# 不写入 data.json；category 仅用于打印分布，也不写入最终结果。
BENCHMARK_NAME = "MathVista"
# 固定扩容前的顺序参考配额，仅用于保留原有50条SAE样本，不随eval规模变化。
SAE_REFERENCE_SAMPLE_COUNT = 200
DATA_DIR = Path("data")


def find_testmini_parquet(data_dir: Path) -> Path:
    """定位唯一的 testmini Parquet 文件。

    功能与逻辑：
        在 data/ 中只匹配 testmini-*.parquet，要求恰好找到一个文件，避免误用
        无公开答案的完整 test 或在多版本文件并存时静默选错数据。

    输入：
        data_dir：Hugging Face 下载生成的 data/ 目录。

    输出：
        唯一 testmini Parquet 文件的 Path。
    """
    if not data_dir.is_dir():
        raise FileNotFoundError("找不到原始数据目录: {}".format(data_dir))

    parquet_paths = sorted(data_dir.glob("testmini-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("data/ 中找不到 testmini-*.parquet")
    if len(parquet_paths) != 1:
        raise ValueError(
            "data/ 中发现多个 testmini Parquet 文件: {}".format(parquet_paths)
        )

    return parquet_paths[0]


def build_pid_sort_key(original_id: str) -> Tuple[int, int, str, str]:
    """为原始 pid 构造确定性的排序键。

    功能与逻辑：
        纯数字 pid 按整数大小排序；其他 pid 排在其后并按不区分大小写的文本
        顺序排列，确保全量编号在重复运行时保持不变。

    输入：
        original_id：原始 pid 的字符串形式。

    输出：
        可供 sorted 使用的四元组排序键。
    """
    if original_id.isdigit():
        return 0, int(original_id), "", original_id

    return 1, 0, original_id.casefold(), original_id


def answer_to_choice_letter(
    answer: Any,
    choices: List[Any],
    original_id: str,
) -> str:
    """将MathVista多项选择题的答案文本转换成唯一选项字母。

    功能与逻辑：
        分别把官方answer和choices转换为去除首尾空白的字符串，要求答案文本在
        choices中恰好匹配一次，再按原始选项位置转换为A、B、C等字母。选项超过
        26个、答案缺失或出现重复匹配时拒绝猜测并报错。

    输入：
        answer：官方提供的正确选项文本。
        choices：按原始顺序排列的候选选项列表。
        original_id：用于错误信息的原始MathVista样本编号。

    输出：
        与正确选项位置对应的单个大写英文字母。
    """
    if len(choices) > len(string.ascii_uppercase):
        raise ValueError(
            "样本{}的选项数超过可用字母数量".format(original_id)
        )
    normalized_answer = str(answer).strip()
    matching_indices = [
        index
        for index, choice in enumerate(choices)
        if str(choice).strip() == normalized_answer
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            "样本{}的答案文本必须在choices中唯一匹配，实际匹配{}次: {!r}".format(
                original_id,
                len(matching_indices),
                normalized_answer,
            )
        )
    return string.ascii_uppercase[matching_indices[0]]


def load_and_number_records(parquet_path: Path) -> List[Dict[str, Any]]:
    """读取、校验并稳定编号全部 testmini 样本。

    功能与逻辑：
        使用 pyarrow 读取 1,000 条 testmini，校验必需字段、pid 唯一性、query、
        答案、metadata 和图片二进制。随后按原始pid稳定排序，在任何筛选和采样
        之前为全量样本分配固定id与图片名称，并在官方query后追加带独立分隔线的
        共享回答格式区块。

    输入：
        parquet_path：testmini Parquet 文件路径。

    输出：
        已完成全量稳定编号和字段转换的样本列表，内部字段以 _ 开头。
    """
    table = parquet.read_table(parquet_path)
    required_fields = {
        "pid",
        "image",
        "decoded_image",
        "choices",
        "answer",
        "question_type",
        "answer_type",
        "metadata",
        "query",
    }
    missing_fields = required_fields - set(table.column_names)

    if missing_fields:
        raise ValueError("testmini 缺少字段: {}".format(sorted(missing_fields)))

    raw_records = table.to_pylist()
    if not raw_records:
        raise ValueError("testmini 中没有样本")

    seen_original_ids: Set[str] = set()
    for row_number, record in enumerate(raw_records, start=1):
        original_id = str(record["pid"]).strip()

        if not original_id:
            raise ValueError("第 {} 条样本的 pid 为空".format(row_number))
        if original_id in seen_original_ids:
            raise ValueError("发现重复 pid: {}".format(original_id))
        if not str(record["query"]).strip():
            raise ValueError("样本 {} 的 query 为空".format(original_id))
        if not str(record["answer"]).strip():
            raise ValueError("样本 {} 的 answer 为空".format(original_id))
        if not isinstance(record["metadata"], dict):
            raise TypeError("样本 {} 的 metadata 必须是 dict".format(original_id))
        if not isinstance(record["decoded_image"], dict):
            raise TypeError("样本 {} 的 decoded_image 必须是 dict".format(original_id))
        if not isinstance(record["decoded_image"].get("bytes"), bytes):
            raise TypeError("样本 {} 缺少图片二进制数据".format(original_id))
        if not record["decoded_image"]["bytes"]:
            raise ValueError("样本 {} 的图片二进制为空".format(original_id))

        metadata = record["metadata"]
        for metadata_key in ("task", "context", "category", "language"):
            if not str(metadata.get(metadata_key, "")).strip():
                raise ValueError(
                    "样本 {} 的 metadata.{} 为空".format(
                        original_id,
                        metadata_key,
                    )
                )

        image_path = str(record["image"]).replace("\\", "/")
        if not PurePosixPath(image_path).suffix:
            raise ValueError("样本 {} 的图片路径没有扩展名".format(original_id))

        seen_original_ids.add(original_id)

    sorted_records = sorted(
        raw_records,
        key=lambda record: build_pid_sort_key(str(record["pid"]).strip()),
    )
    numbered_records: List[Dict[str, Any]] = []

    for sample_number, record in enumerate(sorted_records, start=1):
        metadata = record["metadata"]
        image_path = str(record["image"]).replace("\\", "/")
        image_suffix = PurePosixPath(image_path).suffix.lower()
        image_name = "{}-img-{:04d}{}".format(
            BENCHMARK_NAME,
            sample_number,
            image_suffix,
        )

        numbered_records.append(
            {
                "id": "{}-{:04d}".format(BENCHMARK_NAME, sample_number),
                "img_path": "img/{}".format(image_name),
                "text_input": append_final_answer_instruction(record["query"]),
                "answer": str(record["answer"]).strip(),
                "original_id": str(record["pid"]).strip(),
                "task": str(metadata["task"]).strip(),
                "context": str(metadata["context"]).strip(),
                "_global_order": sample_number,
                "_question_type": str(record["question_type"]).strip().casefold(),
                "_answer_type": str(record["answer_type"]).strip().casefold(),
                "_answer_text": str(record["answer"]).strip(),
                "_language": str(metadata["language"]).strip().casefold(),
                "_category": str(metadata["category"]).strip(),
                "_choices": record["choices"],
                "_image_name": image_name,
                "_image_bytes": record["decoded_image"]["bytes"],
            }
        )

    return numbered_records


def filter_eligible_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """筛选英文多项选择题并校验筛选后的固定属性。

    功能与逻辑：
        在全量稳定编号完成后，仅保留 question_type=multi_choice 且 language=english
        的记录；随后只保留answer_type=text、choices非空且官方答案文本能在choices
        中唯一定位的记录，将答案转换为选项字母。不能转换成统一可评分类型的记录
        会在采样前按原因过滤并打印数量。

    输入：
        records：已经完成全量稳定编号的 1,000 条 testmini 样本。

    输出：
        满足英文多项选择题条件、答案已转换为选项字母且保留全量编号的样本列表。
    """
    eligible_records = [
        record
        for record in records
        if record["_question_type"] == "multi_choice"
        and record["_language"] == "english"
    ]

    rejected_reasons: Counter[str] = Counter()
    converted_records: List[Dict[str, Any]] = []
    for record in eligible_records:
        if record["_answer_type"] != "text":
            rejected_reasons["answer_type不是text"] += 1
            continue
        if not isinstance(record["_choices"], list) or not record["_choices"]:
            rejected_reasons["choices为空或不是list"] += 1
            continue
        try:
            record["answer"] = answer_to_choice_letter(
                record["_answer_text"],
                record["_choices"],
                record["original_id"],
            )
        except ValueError:
            rejected_reasons["答案文本不能唯一映射到选项"] += 1
            continue
        if classify_answer(record["answer"]) != "choice":
            rejected_reasons["转换结果不是单字母选项"] += 1
            continue
        converted_records.append(record)

    print("排除无法规则评分的记录: {}".format(sum(rejected_reasons.values())))
    for reason, count in sorted(rejected_reasons.items()):
        print("  {}: {}".format(reason, count))
    if not converted_records:
        raise ValueError("筛选并转换答案后没有可评分的英文多项选择题")
    return converted_records


def allocate_balanced_quotas(
    group_sizes: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """在若干分组之间容量感知地均衡分配配额。

    功能与逻辑：
        按分组名称固定顺序轮流增加一个配额，达到自身容量的分组停止增加，剩余
        配额继续分给其他分组，直到达到目标总数。容量允许时各组配额之差不超过1。

    输入：
        group_sizes：分组名称到可用样本数量的映射。
        target_count：需要在这些分组间分配的总配额。

    输出：
        分组名称到实际采样配额的映射。
    """
    if target_count < 0:
        raise ValueError("target_count 不能为负数")
    if target_count > sum(group_sizes.values()):
        raise ValueError("目标采样数超过符合条件的样本总数")

    quotas = {group_name: 0 for group_name in group_sizes}
    ordered_group_names = sorted(group_sizes)
    allocated_count = 0

    while allocated_count < target_count:
        allocation_happened = False

        for group_name in ordered_group_names:
            if allocated_count >= target_count:
                break
            if quotas[group_name] >= group_sizes[group_name]:
                continue

            quotas[group_name] += 1
            allocated_count += 1
            allocation_happened = True

        if not allocation_happened:
            raise ValueError("无法完成均衡采样配额分配")

    return quotas


def sample_records(
    records: List[Dict[str, Any]],
    target_sample_count: Optional[int],
    sampling_mode: Literal["sequential", "random"],
) -> List[Dict[str, Any]]:
    """按 task 和 context 两级执行容量感知的均衡采样。

    功能与逻辑：
        先将目标总数尽量平均分配给 5 个 task；容量不足的 task 保留全部，剩余
        配额转给其他 task。随后把每个 task 的配额以相同规则分配给其 context。
        顺序模式取各组全量编号靠前的数据；随机模式无固定种子随机抽取。最终按
        全量编号恢复顺序。

    输入：
        records：筛选后的英文多项选择题列表。
        target_sample_count：目标样本总数；默认 200，None 表示保留全部筛选结果。
        sampling_mode：仅允许 "sequential" 或 "random"。

    输出：
        两级均衡采样后按全量编号排序的样本列表。
    """
    if sampling_mode not in {"sequential", "random"}:
        raise ValueError("sampling_mode 只能是 'sequential' 或 'random'")
    if target_sample_count is None:
        return list(records)
    if target_sample_count <= 0:
        raise ValueError("target_sample_count 必须为正整数或 None")
    if target_sample_count > len(records):
        raise ValueError("target_sample_count 超过筛选后的样本数")

    task_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_groups[record["task"]].append(record)

    task_quotas = allocate_balanced_quotas(
        {task: len(task_records) for task, task_records in task_groups.items()},
        target_sample_count,
    )
    sampled_records: List[Dict[str, Any]] = []

    for task in sorted(task_groups):
        context_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in task_groups[task]:
            context_groups[record["context"]].append(record)

        context_quotas = allocate_balanced_quotas(
            {
                context: len(context_records)
                for context, context_records in context_groups.items()
            },
            task_quotas[task],
        )

        for context in sorted(context_groups):
            context_records = context_groups[context]
            context_quota = context_quotas[context]

            if sampling_mode == "sequential":
                selected_records = context_records[:context_quota]
            else:
                selected_records = random.sample(context_records, k=context_quota)

            sampled_records.extend(selected_records)

    return sorted(sampled_records, key=lambda record: record["_global_order"])


def print_distribution(records: List[Dict[str, Any]], title: str) -> None:
    """打印筛选字段及 task、context、category 分布。

    功能与逻辑：
        统计 question_type、language、answer_type、category、task 及 task/context
        组合，分别用于核对全量数据、筛选结果和最终采样结果。

    输入：
        records：需要统计的样本列表。
        title：当前统计块标题。

    输出：
        无返回值；将分布直接打印到终端。
    """
    question_type_counter = Counter(record["_question_type"] for record in records)
    language_counter = Counter(record["_language"] for record in records)
    answer_type_counter = Counter(record["_answer_type"] for record in records)
    category_counter = Counter(record["_category"] for record in records)
    task_counter = Counter(record["task"] for record in records)
    task_context_counter = Counter(
        (record["task"], record["context"]) for record in records
    )

    print("\n{} {} {}".format("=" * 20, title, "=" * 20))
    print("样本数: {}".format(len(records)))
    print("question_type: {}".format(dict(sorted(question_type_counter.items()))))
    print("language: {}".format(dict(sorted(language_counter.items()))))
    print("answer_type: {}".format(dict(sorted(answer_type_counter.items()))))
    print("category: {}".format(dict(sorted(category_counter.items()))))
    print("task 分布:")
    for task, count in sorted(task_counter.items()):
        print("  {}: {}".format(task, count))

    print("task / context 分布:")
    for combination, count in sorted(task_context_counter.items()):
        print("  {} / {}: {}".format(combination[0], combination[1], count))


def prepare_output_dir(output_dir: str) -> Path:
    """安全清除旧转换结果并创建新的 img/ 目录。

    功能与逻辑：
        限制输出目录必须位于当前 MathVista 根目录内部，并拒绝当前目录、data/
        以及 data/ 的子目录。若输出目录已存在，则完整删除后重新创建 img/。

    输入：
        output_dir：转换结果目录，默认为 "formatted_data"。

    输出：
        新创建的 img/ 目录 Path。
    """
    output_path = Path(output_dir).resolve()
    current_path = Path(".").resolve()
    raw_data_path = DATA_DIR.resolve()

    if output_path == current_path:
        raise ValueError("output_dir 不能是当前 MathVista 根目录")
    if current_path not in output_path.parents:
        raise ValueError("output_dir 必须位于当前 MathVista 目录内部")
    if output_path == raw_data_path or raw_data_path in output_path.parents:
        raise ValueError("output_dir 不能是 data/ 或位于原始 data/ 内部")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("output_dir 已存在但不是目录: {}".format(output_path))

        print("清除上一次转换结果: {}".format(output_path))
        shutil.rmtree(output_path)

    image_dir = output_path / "img"
    image_dir.mkdir(parents=True)
    return image_dir


def write_images(records: List[Dict[str, Any]], image_dir: Path) -> None:
    """将采样样本的图片二进制写入 img/。

    功能与逻辑：
        使用 Parquet 中的 decoded_image.bytes，按全量编号阶段确定的固定文件名
        原样写入；不解压 images.zip，也不创建中间图片目录。

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
        移除所有以下划线开头的筛选与运行时内部字段，仅写入评测和回溯所需的
        公开字段，并记录创建时间与样本数量。

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
    """校验筛选条件、data.json 字段和全部图片路径。

    功能与逻辑：
        确认输出前记录均为英文多项选择题，且答案文本已唯一转换为正确选项字母；
        再重新读取data.json，核对数量、唯一ID、必需字段、内部字段清理、图片
        存在性及img/文件数。

    输入：
        records：写入前的评测样本。
        sae_records：写入前的SAE专用样本。
        output_dir：转换结果目录。

    输出：
        无返回值；校验失败时抛出异常，成功时打印摘要。
    """
    for record in records:
        if record["_question_type"] != "multi_choice":
            raise ValueError("最终样本中存在非多项选择题")
        if record["_language"] != "english":
            raise ValueError("最终样本中存在非英文题目")
        expected_letter = answer_to_choice_letter(
            record["_answer_text"],
            record["_choices"],
            record["original_id"],
        )
        if record["answer"] != expected_letter:
            raise ValueError(
                "样本{}的选项字母答案错误".format(record["original_id"])
            )

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
    target_sample_count: Optional[int] = 200,
    sampling_mode: Literal["sequential", "random"] = "sequential",
    print_distribution_info: bool = True,
) -> None:
    """完成 MathVista/testmini 英文选择题子集的格式转换。

    功能与逻辑：
        先安全清除旧输出，再读取并稳定编号全部 1,000 条 testmini；之后筛选
        英文多项选择题，先固定划出50条SAE专用样本，再从余下记录按task和
        context两级容量感知地均衡采样评测题。写出两组图片并校验互斥ID。

    输入：
        output_dir：输出目录，默认为 "formatted_data"。
        target_sample_count：目标样本总数，默认 200；None 表示保留全部非SAE合格题（按478条容量最多428条）
            符合筛选条件的样本。
        sampling_mode：仅允许 "sequential" 或 "random"。
        print_distribution_info：是否打印全量、筛选后和采样后的分布。

    输出：
        无返回值；在 output_dir 中生成 data.json 和 img/。
    """
    image_dir = prepare_output_dir(output_dir)
    parquet_path = find_testmini_parquet(DATA_DIR)
    all_records = load_and_number_records(parquet_path)
    eligible_records = filter_eligible_records(all_records)
    reference_eval_records = sample_records(
        eligible_records, SAE_REFERENCE_SAMPLE_COUNT, "sequential"
    )
    sae_records = select_sae_records(eligible_records, reference_eval_records)
    sae_ids = {record["id"] for record in sae_records}
    eval_candidates = [record for record in eligible_records if record["id"] not in sae_ids]
    sampled_records = sample_records(
        eval_candidates,
        target_sample_count,
        sampling_mode,
    )

    if print_distribution_info:
        print_distribution(all_records, "全量 testmini")
        print_distribution(eligible_records, "英文多项选择题")
        print_distribution(sampled_records, "采样后数据")

    write_images(sampled_records + sae_records, image_dir)
    write_data_json(sampled_records, sae_records, image_dir.parent)
    validate_output(sampled_records, sae_records, image_dir.parent)


if __name__ == "__main__":
    # 转换结果目录；重复运行时先清除此目录，不会修改 data/ 或 images.zip。
    OUTPUT_DIR = "formatted_data"

    # 默认从英文多项选择题中两级均衡采样 200 条；None 表示保留全部非SAE合格题（按478条容量最多428条）。
    TARGET_SAMPLE_COUNT = 200

    # 可选值："sequential" 或 "random"；随机模式不设置随机种子。
    SAMPLING_MODE = "sequential"

    # True：打印全量、筛选后及采样后分布；False：不打印。
    PRINT_DISTRIBUTION_INFO = True

    convert_dataset(
        output_dir=OUTPUT_DIR,
        target_sample_count=TARGET_SAMPLE_COUNT,
        sampling_mode=SAMPLING_MODE,
        print_distribution_info=PRINT_DISTRIBUTION_INFO,
    )
