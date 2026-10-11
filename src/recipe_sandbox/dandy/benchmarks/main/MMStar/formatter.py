"""将MMStar TSV转换为DataRecipe统一benchmark格式。

脚本读取并校验TSV，按原始index稳定编号，过滤无法用统一规则评分的答案后按
category均衡抽样，并解码Base64图片；将唯一图片写入formatted_data/img并生成
data.json。每条原始题目与选项后用独立分隔区块追加共享回答格式指令；完成后
核对层级分布、答案、字段、ID和图片引用。
"""

import base64
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from recipe_sandbox.dandy.benchmarks.sampling import balanced_eval_records

from recipe_sandbox.dandy.data.answer import (  # noqa: E402
    append_final_answer_instruction,
    partition_supported_answers,
)


# MMStar 共 1,500 条样本，用 category 表示 6 个一级核心能力，每类均为 250 条：
# - coarse perception（粗粒度感知）：整体场景、主题、情绪和图像风格判断；
# - fine-grained perception（细粒度感知）：目标定位、计数和识别；
# - instance reasoning（实例推理）：单实例推理及跨实例属性、关系推理；
# - logical reasoning（逻辑推理）：代码与序列、常识和图示推理；
# - math（数学）：几何、数值常识与计算、统计推理；
# - science & technology（科学与技术）：自然科学及工程、地球与农业知识。
#
# l2_category 表示上述能力下的 18 个细分类别，括号内为全量样本数：
# - coarse perception：image emotion（31）、image scene and topic（141）、
#   image style & quality（78）；
# - fine-grained perception：localization（40）、object counting（92）、
#   recognition（118）；
# - instance reasoning：cross-instance attribute reasoning（89）、
#   cross-instance relation reasoning（62）、single-instance reasoning（99）；
# - logical reasoning：code & sequence reasoning（39）、common reasoning（101）、
#   diagram reasoning（110）；
# - math：geometry（116）、numeric commonsense and calculation（48）、
#   statistical reasoning（86）；
# - science & technology：biology & chemistry & physics（146）、
#   electronics & energy & mechanical eng.（46）、
#   geography & earth science & agriculture（58）。
# 当前采样先保证每个 category 数量相同，再在其 3 个 l2_category 间尽量
# 平均分配；l2_category 只用于内部采样和分布核对，不写入 data.json。
BENCHMARK_NAME = "MMStar"
# 固定扩容前的顺序参考配额，仅用于保留原有50条SAE样本，不随eval规模变化。
SAE_REFERENCE_SAMPLE_COUNT = 33
TSV_PATH = Path("MMStar.tsv")


def normalize_base64_payload(image_value: str) -> str:
    """规范化 TSV 中的 Base64 图片字符串。

    功能与逻辑：
        去除字符串首尾及内部空白；若图片使用 data URL 表示，则移除逗号前的
        媒体类型前缀，仅保留 Base64 正文。

    输入：
        image_value：TSV 的 image 字段。

    输出：
        可直接交给 Base64 解码器的字符串。
    """
    stripped_value = image_value.strip()

    if stripped_value.startswith("data:"):
        if "," not in stripped_value:
            raise ValueError("图片 data URL 缺少逗号分隔符")
        stripped_value = stripped_value.split(",", maxsplit=1)[1]

    return "".join(stripped_value.split())


def decode_image(image_value: str) -> bytes:
    """将 Base64 图片字段解码为二进制数据。

    功能与逻辑：
        先规范化 Base64 字符串，再执行严格解码，并拒绝空图片。

    输入：
        image_value：TSV 的原始 image 字段。

    输出：
        解码后的图片二进制数据。
    """
    payload = normalize_base64_payload(image_value)

    if not payload:
        raise ValueError("发现空的 image 字段")

    image_bytes = base64.b64decode(payload, validate=True)

    if not image_bytes:
        raise ValueError("Base64 解码后图片为空")

    return image_bytes


def detect_image_extension(image_bytes: bytes) -> str:
    """根据文件头判断图片扩展名。

    功能与逻辑：
        识别 MMStar 中常见的 JPEG、PNG、GIF、BMP、WebP 和 TIFF 文件头，
        避免依赖 Pillow 等第三方包。

    输入：
        image_bytes：解码后的图片二进制数据。

    输出：
        包含点号的小写图片扩展名。
    """
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"

    raise ValueError(
        "无法识别图片格式，文件头为: {}".format(image_bytes[:16].hex())
    )


def build_index_sort_key(index_value: str) -> Tuple[int, int, str, str]:
    """为原始 index 构造确定性的排序键。

    功能与逻辑：
        纯数字 index 按整数大小排序；非纯数字 index 排在其后并按不区分大小写
        的文本顺序排列，从而保证重复运行时全量编号顺序不变。

    输入：
        index_value：原始样本 index 的字符串形式。

    输出：
        可供 sorted 使用的四元组排序键。
    """
    if index_value.isdigit():
        return 0, int(index_value), "", index_value

    return 1, 0, index_value.casefold(), index_value


def load_raw_records(tsv_path: Path) -> List[Dict[str, str]]:
    """读取并校验 MMStar.tsv 的全量标注。

    功能与逻辑：
        使用 Python 标准库读取 TSV，避免 pandas 将选项文本 "None" 错当成空值。
        校验必需字段、样本数、index 唯一性以及核心文本字段是否为空。图片内容
        的 Base64 与格式校验在全量稳定编号阶段完成。

    输入：
        tsv_path：MMStar.tsv 的路径。

    输出：
        通过基础字段校验的原始样本列表。
    """
    if not tsv_path.is_file():
        raise FileNotFoundError("找不到 MMStar 标注文件: {}".format(tsv_path))

    csv.field_size_limit(sys.maxsize)
    required_fields = {
        "index",
        "question",
        "answer",
        "category",
        "l2_category",
        "bench",
        "image",
    }

    with tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError("MMStar.tsv 没有表头")

        missing_fields = required_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError("MMStar.tsv 缺少字段: {}".format(sorted(missing_fields)))

        raw_records = list(reader)

    if not raw_records:
        raise ValueError("MMStar.tsv 中没有样本")

    seen_indices: Set[str] = set()
    for row_number, record in enumerate(raw_records, start=2):
        index_value = record["index"].strip()

        if not index_value:
            raise ValueError("TSV 第 {} 行的 index 为空".format(row_number))
        if index_value in seen_indices:
            raise ValueError("发现重复 index: {}".format(index_value))
        if not record["question"].strip():
            raise ValueError("样本 {} 的 question 为空".format(index_value))
        if not record["answer"].strip():
            raise ValueError("样本 {} 的 answer 为空".format(index_value))
        if not record["category"].strip():
            raise ValueError("样本 {} 的 category 为空".format(index_value))
        if not record["l2_category"].strip():
            raise ValueError("样本 {} 的 l2_category 为空".format(index_value))
        if not record["image"].strip():
            raise ValueError("样本 {} 的 image 为空".format(index_value))

        seen_indices.add(index_value)

    return raw_records


def build_numbered_records(
    raw_records: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """对全量样本稳定排序编号，并为唯一图片分配固定名称。

    功能与逻辑：
        先按原始 index 的确定性顺序排列全部样本，再解码图片并计算 SHA-256。
        内容相同的图片共用同一个哈希和输出路径；全部唯一哈希排序后获得固定图片
        编号。最后为1,500条全量样本编号，并在原始question后追加带独立分隔线的
        共享回答格式区块；后续采样不会改变这些编号或text_input。

    输入：
        raw_records：读取并完成基础校验的全部 TSV 样本。

    输出：
        已完成全量稳定编号和字段转换的样本列表，内部字段以 _ 开头。
    """
    sorted_records = sorted(
        raw_records,
        key=lambda record: build_index_sort_key(record["index"].strip()),
    )

    image_extensions: Dict[str, str] = {}
    record_image_info: List[Tuple[str, str]] = []

    for record in sorted_records:
        image_bytes = decode_image(record["image"])
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_extension = detect_image_extension(image_bytes)

        if image_hash in image_extensions:
            if image_extensions[image_hash] != image_extension:
                raise ValueError("同一图片哈希对应了不同扩展名: {}".format(image_hash))
        else:
            image_extensions[image_hash] = image_extension

        record_image_info.append((image_hash, image_extension))

    image_name_map: Dict[str, str] = {}
    for image_number, image_hash in enumerate(sorted(image_extensions), start=1):
        image_name_map[image_hash] = "{}-img-{:04d}{}".format(
            BENCHMARK_NAME,
            image_number,
            image_extensions[image_hash],
        )

    numbered_records: List[Dict[str, Any]] = []
    for sample_number, (record, image_info) in enumerate(
        zip(sorted_records, record_image_info),
        start=1,
    ):
        image_hash, image_extension = image_info
        numbered_records.append(
            {
                "id": "{}-{:04d}".format(BENCHMARK_NAME, sample_number),
                "img_path": "img/{}".format(image_name_map[image_hash]),
                "text_input": append_final_answer_instruction(record["question"]),
                "answer": record["answer"].strip(),
                "original_id": record["index"].strip(),
                "category": record["category"].strip(),
                "_global_order": sample_number,
                "_l2_category": record["l2_category"].strip(),
                "_image_hash": image_hash,
                "_image_extension": image_extension,
                "_image_base64": record["image"],
                "_bench": record["bench"].strip(),
            }
        )

    return numbered_records


def allocate_balanced_quotas(
    group_sizes: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """在若干二级类别之间尽量平均分配采样配额。

    功能与逻辑：
        按二级类别名称的固定顺序轮流增加一个配额，直到配额总和达到目标数量；
        已达到自身样本容量的类别不再增加，剩余配额继续分给其他类别。因此在
        数据容量允许时，各二级类别配额之差不超过 1。

    输入：
        group_sizes：二级类别名称到可用样本数的映射。
        target_count：当前一级 category 需要采样的总数。

    输出：
        二级类别名称到实际采样配额的映射。
    """
    if target_count < 0:
        raise ValueError("target_count 不能为负数")
    if target_count > sum(group_sizes.values()):
        raise ValueError("目标数量超过当前 category 的可用样本数")

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
            raise ValueError("无法完成二级类别采样配额分配")

    return quotas


def sample_records_by_category(
    records: List[Dict[str, Any]],
    samples_per_category: Optional[int],
    sampling_mode: Literal["sequential", "random"],
) -> List[Dict[str, Any]]:
    """按每个 category 相同数量采样。

    功能与逻辑：
        全量样本已经稳定编号后，先按 category 分组，再将每个 category 的采样
        数量在其所属的 l2_category 间尽量平均分配。顺序模式在每个二级类别内取
        编号最靠前的记录；随机模式在每个二级类别内无固定种子随机抽取。所有
        采样结果最后恢复为全量编号顺序，且保留原来的 id。

    输入：
        records：已经完成全量稳定编号的样本。
        samples_per_category：每个 category 的采样数；None 表示保留全部输入记录。
        sampling_mode：仅允许 "sequential" 或 "random"。

    输出：
        按 category 等量采样并恢复稳定顺序后的样本列表。
    """
    if sampling_mode not in {"sequential", "random"}:
        raise ValueError(
            "sampling_mode 只能是 'sequential' 或 'random': {}".format(
                sampling_mode
            )
        )

    if samples_per_category is None:
        return list(records)

    if samples_per_category <= 0:
        raise ValueError("samples_per_category 必须为正整数或 None")

    grouped_records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[record["category"]].append(record)

    sampled_records: List[Dict[str, Any]] = []
    for category in sorted(grouped_records):
        category_records = grouped_records[category]

        if samples_per_category > len(category_records):
            raise ValueError(
                "category '{}' 只有 {} 条，无法采样 {} 条".format(
                    category,
                    len(category_records),
                    samples_per_category,
                )
            )

        l2_grouped_records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in category_records:
            l2_grouped_records[record["_l2_category"]].append(record)

        l2_quotas = allocate_balanced_quotas(
            {
                l2_category: len(l2_records)
                for l2_category, l2_records in l2_grouped_records.items()
            },
            samples_per_category,
        )

        for l2_category in sorted(l2_grouped_records):
            l2_records = l2_grouped_records[l2_category]
            l2_quota = l2_quotas[l2_category]

            if sampling_mode == "sequential":
                selected_records = l2_records[:l2_quota]
            else:
                selected_records = random.sample(l2_records, k=l2_quota)

            sampled_records.extend(selected_records)

    return sorted(sampled_records, key=lambda record: record["_global_order"])


def print_distribution(records: List[Dict[str, Any]], title: str) -> None:
    """打印样本的能力类别、细分类别、来源和答案分布。

    功能与逻辑：
        分别统计 category、l2_category、二者组合、bench 和 answer，帮助核对
        全量数据以及 category 等量采样后的实际构成。

    输入：
        records：需要统计的样本列表。
        title：当前统计块的标题。

    输出：
        无返回值；将分布信息打印到终端。
    """
    category_counter = Counter(record["category"] for record in records)
    l2_category_counter = Counter(record["_l2_category"] for record in records)
    combination_counter = Counter(
        (record["category"], record["_l2_category"]) for record in records
    )
    bench_counter = Counter(record["_bench"] for record in records)
    answer_counter = Counter(record["answer"] for record in records)

    print("\n{}：{} 条样本，{} 张唯一图片".format(
        title,
        len(records),
        len({record["_image_hash"] for record in records}),
    ))
    print("Category 分布:")
    for category, count in sorted(category_counter.items()):
        print("  {}: {}".format(category, count))

    print("L2 Category 分布:")
    for l2_category, count in sorted(l2_category_counter.items()):
        print("  {}: {}".format(l2_category, count))

    print("Category / L2 Category 分布:")
    for combination, count in sorted(combination_counter.items()):
        print("  {} / {}: {}".format(combination[0], combination[1], count))

    print("Bench 来源分布:")
    for bench, count in sorted(bench_counter.items()):
        print("  {}: {}".format(bench, count))

    print("Answer 分布:")
    for answer, count in sorted(answer_counter.items()):
        print("  {}: {}".format(answer, count))


def prepare_output_dir(output_dir: str) -> Path:
    """安全清除旧转换结果并创建新的 img/ 目录。

    功能与逻辑：
        限制输出目录必须位于当前 MMStar 目录内部，且不能是当前目录或原始 TSV。
        若输出目录已存在，则完整删除上一次转换结果，再创建新的 img/。

    输入：
        output_dir：转换结果目录，默认为 "formatted_data"。

    输出：
        新创建的 img/ 目录 Path。
    """
    output_path = Path(output_dir).resolve()
    current_path = Path(".").resolve()
    source_tsv_path = TSV_PATH.resolve()

    if output_path == current_path:
        raise ValueError("output_dir 不能是当前 MMStar 根目录")
    if current_path not in output_path.parents:
        raise ValueError("output_dir 必须位于当前 MMStar 目录内部")
    if output_path == source_tsv_path:
        raise ValueError("output_dir 不能指向原始 MMStar.tsv")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("output_dir 已存在但不是目录: {}".format(output_path))

        print("清除上一次转换结果: {}".format(output_path))
        shutil.rmtree(output_path)

    image_dir = output_path / "img"
    image_dir.mkdir(parents=True)
    return image_dir


def write_selected_images(
    records: List[Dict[str, Any]],
    image_dir: Path,
) -> int:
    """把采样样本引用的唯一 Base64 图片写入 img/。

    功能与逻辑：
        按样本稳定顺序遍历，只对尚未写出的图片哈希执行一次解码和写入；写入前
        重新核对哈希及扩展名，确保最终图片内容与全量编号阶段一致。

    输入：
        records：采样后的样本列表。
        image_dir：最终图片输出目录。

    输出：
        实际写出的唯一图片数量。
    """
    written_hashes: Set[str] = set()

    for record in records:
        image_hash = record["_image_hash"]
        if image_hash in written_hashes:
            continue

        image_bytes = decode_image(record["_image_base64"])
        calculated_hash = hashlib.sha256(image_bytes).hexdigest()
        calculated_extension = detect_image_extension(image_bytes)

        if calculated_hash != image_hash:
            raise ValueError("样本 {} 的图片哈希发生变化".format(record["original_id"]))
        if calculated_extension != record["_image_extension"]:
            raise ValueError("样本 {} 的图片格式发生变化".format(record["original_id"]))

        output_name = Path(record["img_path"]).name
        (image_dir / output_name).write_bytes(image_bytes)
        written_hashes.add(image_hash)

    return len(written_hashes)


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
        移除所有以下划线开头的运行时内部字段，仅保留评测和回溯所需字段；随后
        写入创建时间、样本数量和样本数组。

    输入：
        records：评测样本列表。
        sae_records：与评测ID互斥的SAE专用样本列表。
        output_dir：包含 data.json 和 img/ 的输出目录。

    输出：
        无返回值；在 output_dir 中生成 data.json。
    """
    public_records = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
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

    data_json_path = output_dir / "data.json"
    data_json_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_output(
    records: List[Dict[str, Any]],
    sae_records: List[Dict[str, Any]],
    expected_image_count: int,
    output_dir: Path,
) -> None:
    """校验转换结果的数量、字段和图片路径。

    功能与逻辑：
        重新读取 data.json，核对顶层数量、样本必需字段、唯一 id、内部字段清理、
        img_path 文件存在性以及 img/ 内实际文件数。

    输入：
        records：写入前的评测样本列表。
        sae_records：写入前的SAE专用样本列表。
        expected_image_count：预期唯一图片数量。
        output_dir：转换结果目录。

    输出：
        无返回值；校验失败时抛出明确异常，成功时打印摘要。
    """
    data_json_path = output_dir / "data.json"
    saved_data = json.loads(data_json_path.read_text(encoding="utf-8"))
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
        if any(key.startswith("_") for key in record):
            raise ValueError("data.json 中仍包含内部字段")
        if record["id"] in saved_ids:
            raise ValueError("data.json 中存在重复 id: {}".format(record["id"]))
        if not (output_dir / record["img_path"]).is_file():
            raise FileNotFoundError("输出图片不存在: {}".format(record["img_path"]))

        saved_ids.add(record["id"])

    output_images = [path for path in (output_dir / "img").iterdir() if path.is_file()]
    if len(output_images) != expected_image_count:
        raise ValueError(
            "img/ 文件数 {} 与预期唯一图片数 {} 不一致".format(
                len(output_images),
                expected_image_count,
            )
        )

    print("\n转换结果校验通过")
    print("样本数: {}".format(len(saved_records)))
    print("SAE专用样本数: {}".format(len(saved_sae_records)))
    print("唯一图片数: {}".format(expected_image_count))
    print("输出目录: {}".format(output_dir))


def convert_dataset(
    output_dir: str = "formatted_data",
    samples_per_category: Optional[int] = None,
    sampling_mode: Literal["sequential", "random"] = "sequential",
    print_distribution_info: bool = True,
    target_sample_count: Optional[int] = 200,
) -> None:
    """完成 MMStar 的 category 等量采样与格式转换。

    功能与逻辑：
        首先安全清除旧输出，再读取全部1,500条TSV数据并进行稳定排序、样本编号
        和唯一图片编号；使用统一答案规则排除无法评分的记录，先固定划出50条
        SAE专用样本，再按每个category相同数量采样剩余记录，并在每个category内
        对l2_category尽量平均分配。最后写出两组记录引用的唯一图片和data.json。

    输入：
        target_sample_count：排除SAE后的eval总数，默认200；None取消总量限制。
        output_dir：输出目录，默认为 "formatted_data"。
        samples_per_category：每个 category 的采样数；默认None，不限制每组数量；
            None 表示保留全部非SAE合格题。
        sampling_mode：仅允许 "sequential" 或 "random"。
        print_distribution_info：是否打印全量和采样后的分布。

    输出：
        无返回值；在 output_dir 中生成 data.json 和 img/。
    """
    image_dir = prepare_output_dir(output_dir)
    raw_records = load_raw_records(TSV_PATH)
    numbered_records = build_numbered_records(raw_records)
    all_records, rejected_records = partition_supported_answers(numbered_records)
    print("排除无法规则评分的记录: {}".format(len(rejected_records)))
    reference_eval_records = sample_records_by_category(
        all_records, SAE_REFERENCE_SAMPLE_COUNT, "sequential"
    )
    sae_records = select_sae_records(all_records, reference_eval_records)
    sae_ids = {record["id"] for record in sae_records}
    eval_candidates = [record for record in all_records if record["id"] not in sae_ids]
    sampled_records = sample_records_by_category(
        eval_candidates,
        samples_per_category,
        sampling_mode,
    )
    sampled_records = balanced_eval_records(
        sampled_records, target_sample_count, (('category',), ('_l2_category',)), sampling_mode
    )

    if print_distribution_info:
        print_distribution(all_records, "全量数据")
        print_distribution(sampled_records, "采样后数据")

    expected_image_count = write_selected_images(sampled_records + sae_records, image_dir)
    write_data_json(sampled_records, sae_records, image_dir.parent)
    validate_output(sampled_records, sae_records, expected_image_count, image_dir.parent)


if __name__ == "__main__":
    # 转换结果目录；重复运行时先清除此目录，不会修改 MMStar.tsv 等原始文件。
    OUTPUT_DIR = "formatted_data"

    # 不限制组容量；下方总量200按原分层均衡分配。
    SAMPLES_PER_CATEGORY = None
    TARGET_SAMPLE_COUNT = 200

    # 可选值："sequential" 或 "random"；随机模式不设置随机种子。
    SAMPLING_MODE = "sequential"

    # True：打印全量及采样后分布；False：不打印。
    PRINT_DISTRIBUTION_INFO = True

    convert_dataset(
        output_dir=OUTPUT_DIR,
        samples_per_category=SAMPLES_PER_CATEGORY,
        sampling_mode=SAMPLING_MODE,
        print_distribution_info=PRINT_DISTRIBUTION_INFO,
        target_sample_count=TARGET_SAMPLE_COUNT,
    )
