"""将官方MME-RealWorld-Lite转换为DataRecipe统一OOD benchmark格式。

脚本直接读取标注与图片压缩包，规范图片路径并稳定编号，过滤无法用统一规则评分
的答案后按Subtask与Category均衡抽取快速评测子集。原始题目和候选项后用独立
分隔区块追加共享回答格式指令；选中图片写入formatted_data/img，标注写入
data.json，最后核对分布、字段、ID和图片引用。
"""

import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from recipe_sandbox.dandy.data.answer import (  # noqa: E402
    append_final_answer_instruction,
    partition_supported_answers,
)


# MME-RealWorld-Lite 共 1,919 条 QA，使用 Task、Subtask 和 Category
# 描述三级评测结构：
# - Task=Perception（感知，1,169 条）：直接从图像中识别目标、属性、文字、
#   数量或位置；
# - Task=Reasoning（推理，750 条）：根据图像进行关系、意图、计算或场景推理。
#
# Subtask 表示 5 类真实场景，Category 表示场景下的具体能力：
# 1. Autonomous_Driving（自动驾驶，750 条，15 类，每类 50 条）
#    感知类：Attribute_Motion_MultiPedestrians、
#    Attribute_Motion_MultiVehicles、Attribute_Motion_Pedestrain、
#    Attribute_Motion_Vehicle、Attribute_Visual_TrafficSignal、
#    Object_Count、Objects_Identify。
#    推理类：Attention_TrafficSignal、Prediction_Intention_Ego、
#    Prediction_Intention_Pedestrian、Prediction_Intention_Vehicle、
#    Relation_Interaction_Ego2Pedestrain、
#    Relation_Interaction_Ego2TrafficSignal、
#    Relation_Interaction_Ego2Vehicle、
#    Relation_Interaction_Other2Other。
#
# 2. Diagram and Table（图表，200 条，2 类）
#    diagram（100）、table（100）；两类均同时包含感知与推理问题。
#
# 3. Monitoring（监控场景，469 条，12 类）
#    感知类：Person/counting（1）、Vehicle/counting（19）、
#    person/attribute/color（50）、person/attribute/orientation（19）、
#    person/counting（49）、vehicle/attribute/color（50）、
#    vehicle/attribute/orientation（50）、vehicle/counting（31）、
#    vehicle/location（50）。
#    推理类：calculate（50）、intention（50）、property（50）。
#    注意 Person/counting、Vehicle/counting 与对应的小写名称是不同标签。
#
# 4. OCR with Complex Context（复杂场景 OCR，350 条，7 类，每类 50 条）
#    感知类：adver_and_product、book_map_poster、license、
#    phone_and_address、text_recog。
#    推理类：Character Identification、Scene Understanding。
#
# 5. Remote Sensing（遥感，150 条，3 类，每类 50 条）
#    均为感知类：color、count、position。
#
# 当前从官方 Lite 构造约 200 条快速测试子集：先在 Subtask 间均衡分配
# 总配额，再在每个 Subtask 的 Category 间均衡分配。该结果属于
# MME-RealWorld-Lite 自定义子集，不能表述为官方 Lite 完整评测结果。
# 原始 Text 与 Answer choices 共同组成题目区块，随后通过分隔线追加共享回答格式
# 指令并写入text_input；Ground truth是正确选项字母，Dataset字段不写入data.json。

BENCHMARK_NAME = "MME-RealWorld-Lite"
ZIP_PATH = Path("data.zip")
ANNOTATION_MEMBER = "data/MME-RealWorld-Lite.json"
IMAGE_MEMBER_PREFIX = "data/imgs/"


def normalize_image_path(image_path: str) -> str:
    """规范化并校验 JSON 中的图片相对路径。

    功能与逻辑：
        将反斜杠统一为正斜杠，并拒绝绝对路径、空路径和包含父目录跳转的路径，
        防止异常标注访问 data/imgs/ 之外的 ZIP 成员。

    输入：
        image_path：原始 JSON 中 Image 字段的字符串。

    输出：
        经过校验并统一分隔符的 ZIP 内相对图片路径。
    """
    normalized_path = image_path.replace("\\", "/")
    pure_path = PurePosixPath(normalized_path)

    if not normalized_path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError("发现不安全的图片相对路径: {}".format(image_path))

    return pure_path.as_posix()


def load_raw_records(zip_path: Path) -> List[Dict[str, Any]]:
    """从 data.zip 中读取并校验官方 Lite 标注。

    功能与逻辑：
        直接读取 ZIP 内的 data/MME-RealWorld-Lite.json，不将 ZIP 全量解压。
        校验根对象类型、每条样本的必需字段、选项格式以及图片成员是否存在。
        Lite 中 Question_id 可能重复，因此记录每条数据在原始 JSON 中的行号，
        作为后续唯一回溯依据。

    输入：
        zip_path：官方 Lite 的 data.zip 路径。

    输出：
        通过字段和图片存在性校验的原始样本字典列表。
    """
    if not zip_path.is_file():
        raise FileNotFoundError("找不到官方 Lite 压缩包: {}".format(zip_path))

    required_fields = {
        "Question_id",
        "Image",
        "Text",
        "Answer choices",
        "Ground truth",
        "Category",
        "Subtask",
        "Task",
    }

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        archive_members = set(archive.namelist())

        if ANNOTATION_MEMBER not in archive_members:
            raise FileNotFoundError(
                "ZIP 内找不到标注文件: {}".format(ANNOTATION_MEMBER)
            )

        raw_records = json.loads(archive.read(ANNOTATION_MEMBER).decode("utf-8"))

        if not isinstance(raw_records, list):
            raise TypeError("标注文件根对象必须是 list")

        question_id_counter: Counter = Counter()

        for index, record in enumerate(raw_records, start=1):
            missing_fields = required_fields - set(record.keys())

            if missing_fields:
                raise ValueError(
                    "第 {} 条样本缺少字段: {}".format(index, sorted(missing_fields))
                )

            question_id = str(record["Question_id"])
            question_id_counter.update([question_id])
            record["_source_index"] = index

            if not isinstance(record["Answer choices"], list):
                raise TypeError(
                    "样本 {} 的 Answer choices 必须是 list".format(question_id)
                )

            normalized_image_path = normalize_image_path(str(record["Image"]))
            image_member = "{}{}".format(IMAGE_MEMBER_PREFIX, normalized_image_path)

            if image_member not in archive_members:
                raise FileNotFoundError(
                    "样本 {} 对应图片不在 ZIP 中: {}".format(
                        question_id, image_member
                    )
                )

        duplicate_question_ids = {
            question_id: count
            for question_id, count in question_id_counter.items()
            if count > 1
        }

        if duplicate_question_ids:
            duplicate_record_count = sum(
                count - 1 for count in duplicate_question_ids.values()
            )
            print(
                "检测到 {} 个重复 Question_id，共涉及 {} 条额外记录；"
                "将使用 original_index 区分。".format(
                    len(duplicate_question_ids), duplicate_record_count
                )
            )

    if not raw_records:
        raise ValueError("标注文件中没有有效样本")

    return raw_records


def build_numbered_records(
    raw_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """对全量样本稳定排序编号，并建立唯一图片名称映射。

    功能与逻辑：
        先按 Question_id、图片路径、题目、选项、答案和原始 JSON 行号组成的
        复合键稳定排列全部样本，再连续编号。即使 Question_id 重复，编号仍然
        唯一且可复现。同时对全部唯一Image路径稳定排序，为每张图片分配唯一输出
        文件名；多条QA引用同一原图时复用同一个img_path。每条记录的题目和候选项
        组合完成后，再追加带独立分隔线的共享回答格式区块。

    输入：
        raw_records：通过校验的全部原始样本。

    输出：
        二元组：
        1. 已稳定编号并完成字段转换的全部样本列表；
        2. 原始图片相对路径到输出图片文件名的映射。
    """
    sorted_records = sorted(
        raw_records,
        key=lambda record: (
            str(record["Question_id"]).casefold(),
            str(record["Question_id"]),
            normalize_image_path(str(record["Image"])).casefold(),
            normalize_image_path(str(record["Image"])),
            str(record["Text"]),
            json.dumps(record["Answer choices"], ensure_ascii=False),
            str(record["Ground truth"]),
            int(record["_source_index"]),
        ),
    )

    unique_image_paths = sorted(
        {normalize_image_path(str(record["Image"])) for record in sorted_records},
        key=lambda image_path: (image_path.casefold(), image_path),
    )

    image_name_map: Dict[str, str] = {}
    for image_number, image_path in enumerate(unique_image_paths, start=1):
        suffix = PurePosixPath(image_path).suffix.lower()

        if not suffix:
            raise ValueError("图片没有扩展名: {}".format(image_path))

        image_name_map[image_path] = "{}-img-{:04d}{}".format(
            BENCHMARK_NAME, image_number, suffix
        )

    numbered_records: List[Dict[str, Any]] = []

    for sample_number, record in enumerate(sorted_records, start=1):
        original_id = str(record["Question_id"])
        image_path = normalize_image_path(str(record["Image"]))
        answer_choices = [str(choice).strip() for choice in record["Answer choices"]]
        question_text = str(record["Text"]).strip()

        text_input = question_text
        if answer_choices:
            text_input = "{}\n\n{}".format(
                question_text,
                "\n".join(answer_choices),
            )

        numbered_records.append(
            {
                "id": "{}-{:04d}".format(BENCHMARK_NAME, sample_number),
                "img_path": "img/{}".format(image_name_map[image_path]),
                "text_input": append_final_answer_instruction(text_input),
                "answer": str(record["Ground truth"]).strip(),
                "original_id": original_id,
                "original_index": int(record["_source_index"]),
                "task": str(record["Task"]),
                "subtask": str(record["Subtask"]),
                "category": str(record["Category"]),
                "_global_order": sample_number,
                "_source_image_path": image_path,
            }
        )

    return numbered_records, image_name_map


def allocate_balanced_quotas(
    group_sizes: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """在若干分组之间均衡分配指定的样本总配额。

    功能与逻辑：
        按分组名称的固定顺序轮流增加一个配额，直到达到目标总数；已达到自身
        样本容量的分组不再增加，其剩余配额会继续分配给其他分组。

    输入：
        group_sizes：分组名称到该组可用样本数的映射。
        target_count：需要在所有分组间分配的样本总数。

    输出：
        每个分组最终获得的样本配额映射，配额总和不超过 target_count。
    """
    if target_count < 0:
        raise ValueError("target_count 不能小于 0")

    quotas = {group_name: 0 for group_name in group_sizes}
    remaining = min(target_count, sum(group_sizes.values()))
    sorted_group_names = sorted(group_sizes)

    while remaining > 0:
        allocated_in_round = False

        for group_name in sorted_group_names:
            if remaining == 0:
                break

            if quotas[group_name] < group_sizes[group_name]:
                quotas[group_name] += 1
                remaining -= 1
                allocated_in_round = True

        if not allocated_in_round:
            break

    return quotas


def sample_records(
    records: List[Dict[str, Any]],
    target_sample_count: Optional[int],
    sampling_mode: Literal["sequential", "random"],
) -> List[Dict[str, Any]]:
    """按 Subtask 和 Category 两级均衡采样。

    功能与逻辑：
        全量样本已提前完成稳定排序和编号。本函数先将目标总量均衡分配给各
        Subtask，再把每个 Subtask 的配额均衡分配给其 Category。顺序模式
        选择组内原始顺序靠前的样本，随机模式执行无放回随机抽样且不设置
        随机种子。最终按全量编号顺序排列选中样本。

    输入：
        records：已经完成全量稳定编号的样本列表。
        target_sample_count：目标样本总数；None 表示保留全部样本。
        sampling_mode：仅允许 "sequential" 或 "random"。

    输出：
        按全量编号顺序排列的采样后样本列表。
    """
    if target_sample_count is None or target_sample_count >= len(records):
        return records

    if target_sample_count <= 0:
        raise ValueError("target_sample_count 必须为正整数或 None")

    if sampling_mode not in {"sequential", "random"}:
        raise ValueError("sampling_mode 只能是 'sequential' 或 'random'")

    subtask_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        subtask_groups[record["subtask"]].append(record)

    subtask_sizes = {
        subtask: len(group_records)
        for subtask, group_records in subtask_groups.items()
    }
    subtask_quotas = allocate_balanced_quotas(
        subtask_sizes,
        target_sample_count,
    )

    selected_records: List[Dict[str, Any]] = []

    for subtask in sorted(subtask_groups):
        category_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for record in subtask_groups[subtask]:
            category_groups[record["category"]].append(record)

        category_sizes = {
            category: len(group_records)
            for category, group_records in category_groups.items()
        }
        category_quotas = allocate_balanced_quotas(
            category_sizes,
            subtask_quotas[subtask],
        )

        for category in sorted(category_groups):
            group_records = category_groups[category]
            sample_count = category_quotas[category]

            if sampling_mode == "sequential":
                selected_records.extend(group_records[:sample_count])
            else:
                selected_records.extend(random.sample(group_records, sample_count))

    selected_records.sort(key=lambda record: record["_global_order"])

    if len(selected_records) != target_sample_count:
        raise ValueError(
            "采样数量不一致，预期 {}，实际 {}".format(
                target_sample_count, len(selected_records)
            )
        )

    return selected_records


def select_image_name_map(
    records: List[Dict[str, Any]],
    full_image_name_map: Dict[str, str],
) -> Dict[str, str]:
    """从全量图片映射中筛选采样样本实际引用的唯一图片。

    功能与逻辑：
        收集采样记录中的原始图片路径，并从全量稳定图片编号映射中取出对应
        项目，确保只解压入选样本需要的图片，同时保持图片编号不因采样而变化。

    输入：
        records：采样后的样本列表。
        full_image_name_map：全量原始图片路径到稳定输出文件名的映射。

    输出：
        采样样本实际需要的唯一图片名称映射。
    """
    selected_image_paths = {
        record["_source_image_path"]
        for record in records
    }

    return {
        image_path: full_image_name_map[image_path]
        for image_path in sorted(selected_image_paths)
    }


def print_distribution(records: List[Dict[str, Any]], title: str) -> None:
    """打印转换数据的 Task、Subtask 及其 Category 分布。

    功能与逻辑：
        先打印 Task 和 Subtask 的总体样本数，再按 Subtask 分组打印其下全部
        Category 的类别数与样本数，方便核对两级均衡采样结果。

    输入：
        records：待统计的样本列表。
        title：用于区分全量数据和采样后数据的标题。

    输出：
        无返回值；分布信息直接打印到终端。
    """
    print("\n{} {} {}".format("=" * 20, title, "=" * 20))
    task_counter = Counter(record["task"] for record in records)
    subtask_counter = Counter(record["subtask"] for record in records)
    category_groups: Dict[str, Counter] = defaultdict(Counter)

    for record in records:
        category_groups[record["subtask"]].update([record["category"]])

    print("\nTask 分布:")
    for task, count in sorted(task_counter.items()):
        print("  {}: {}".format(task, count))

    print("\nSubtask 分布:")
    for subtask, count in sorted(subtask_counter.items()):
        print("  {}: {}".format(subtask, count))

    for subtask in sorted(category_groups):
        category_counter = category_groups[subtask]
        print(
            "\n{}：{} 个 Category，{} 条样本".format(
                subtask,
                len(category_counter),
                sum(category_counter.values()),
            )
        )

        for category, count in sorted(category_counter.items()):
            print("  {}: {}".format(category, count))


def prepare_output_dir(output_dir: str) -> Path:
    """清除旧转换结果并创建新的 formatted_data/img/。

    功能与逻辑：
        解析输出路径并限制其必须位于当前工作目录内部。若输出目录已存在，
        则删除上一次转换结果，再创建新的 img/ 目录。不会修改 data.zip
        或其他原始文件。

    输入：
        output_dir：转换结果目录，默认应为 "formatted_data"。

    输出：
        新创建的 img/ 目录 Path。
    """
    output_path = Path(output_dir).resolve()
    current_path = Path(".").resolve()
    source_zip_path = ZIP_PATH.resolve()

    if output_path == current_path:
        raise ValueError("output_dir 不能是当前 MME-RealWorld-Lite 根目录")

    if current_path not in output_path.parents:
        raise ValueError("output_dir 必须位于当前 MME-RealWorld-Lite 目录内部")

    if output_path == source_zip_path or source_zip_path in output_path.parents:
        raise ValueError("output_dir 不能指向原始 data.zip 或位于其中")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("output_dir 已存在但不是目录: {}".format(output_path))

        print("清除上一次转换结果: {}".format(output_path))
        shutil.rmtree(output_path)

    image_dir = output_path / "img"
    image_dir.mkdir(parents=True)
    return image_dir


def extract_unique_images(
    zip_path: Path,
    image_name_map: Dict[str, str],
    image_dir: Path,
) -> None:
    """直接从 data.zip 提取并重命名全部唯一图片。

    功能与逻辑：
        不创建中间解压目录；逐张读取 ZIP 中 data/imgs/ 下被标注引用的图片，
        复制到 formatted_data/img/。同一原图只处理一次。

    输入：
        zip_path：官方 Lite 的 data.zip 路径。
        image_name_map：原始图片相对路径到输出文件名的映射。
        image_dir：最终 img/ 输出目录。

    输出：
        无返回值；在 image_dir 中生成全部唯一图片。
    """
    total_images = len(image_name_map)

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        for index, image_path in enumerate(sorted(image_name_map), start=1):
            image_member = "{}{}".format(IMAGE_MEMBER_PREFIX, image_path)
            target_path = image_dir / image_name_map[image_path]

            with archive.open(image_member, mode="r") as source_file:
                with target_path.open("wb") as target_file:
                    shutil.copyfileobj(
                        source_file,
                        target_file,
                        length=16 * 1024 * 1024,
                    )

            if index % 100 == 0 or index == total_images:
                print("已提取图片: {}/{}".format(index, total_images))


def write_data_json(records: List[Dict[str, Any]], output_dir: Path) -> None:
    """按照统一格式写入最终 data.json。

    功能与逻辑：
        将创建时间、样本数和全部转换后样本写入 UTF-8 JSON 文件。输出字段
        不含 Dataset、原始图片文件名或运行时临时路径。

    输入：
        records：已完成排序、编号和字段转换的样本列表。
        output_dir：data.json 所在的输出目录。

    输出：
        无返回值；生成 output_dir/data.json。
    """
    clean_records = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        for record in records
    ]

    output_data = {
        "created_at": datetime.now().astimezone().isoformat(),
        "num_samples": len(clean_records),
        "data": clean_records,
    }

    json_path = output_dir / "data.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("已生成标注文件: {}".format(json_path))


def validate_output(
    records: List[Dict[str, Any]],
    image_name_map: Dict[str, str],
    output_dir: Path,
) -> None:
    """核对最终数据量、唯一图片数和所有 img_path。

    功能与逻辑：
        检查 data.json 是否存在、img/ 中的文件数是否等于唯一原图数，并确认
        每条样本的 img_path 都指向实际文件。发现不一致时立即报错。

    输入：
        records：最终样本列表。
        image_name_map：唯一原始图片到输出文件名的映射。
        output_dir：最终转换结果目录。

    输出：
        无返回值；校验通过时打印最终统计。
    """
    json_path = output_dir / "data.json"
    image_dir = output_dir / "img"

    if not json_path.is_file():
        raise FileNotFoundError("转换后缺少 data.json")

    output_images = [path for path in image_dir.iterdir() if path.is_file()]

    if len(output_images) != len(image_name_map):
        raise ValueError(
            "输出图片数不一致，预期 {}，实际 {}".format(
                len(image_name_map), len(output_images)
            )
        )

    missing_paths = [
        record["img_path"]
        for record in records
        if not (output_dir / record["img_path"]).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "有 {} 条样本的 img_path 不存在".format(len(missing_paths))
        )

    print("\n转换结果校验通过")
    print("QA 数: {}".format(len(records)))
    print("唯一图片数: {}".format(len(output_images)))
    print("输出目录: {}".format(output_dir))


def convert_dataset(
    output_dir: str = "formatted_data",
    target_sample_count: Optional[int] = 200,
    sampling_mode: Literal["sequential", "random"] = "random",
    print_distribution_info: bool = True,
) -> None:
    """完成 MME-RealWorld-Lite 自定义快速子集的格式转换。

    功能与逻辑：
        从data.zip读取和校验全量标注，先稳定排序并编号全部1,919条样本，使用
        统一答案规则排除无法评分的记录，再按Subtask和Category两级均衡采样。
        随后只提取采样记录引用的唯一图片，写入data.json并验证所有输出路径与数量。

    输入：
        output_dir：输出目录，默认为 "formatted_data"。
        target_sample_count：采样目标总数，默认 200；None 表示保留全部。
        sampling_mode：仅允许 "sequential" 或 "random"。
        print_distribution_info：是否打印 Task、Subtask 和 Category 分布。

    输出：
        无返回值；在 output_dir 中生成 data.json 和 img/。
    """
    image_dir = prepare_output_dir(output_dir)
    raw_records = load_raw_records(ZIP_PATH)
    numbered_records, full_image_name_map = build_numbered_records(raw_records)
    all_records, rejected_records = partition_supported_answers(numbered_records)
    print("排除无法规则评分的记录: {}".format(len(rejected_records)))
    records = sample_records(
        all_records,
        target_sample_count,
        sampling_mode,
    )
    image_name_map = select_image_name_map(records, full_image_name_map)

    if print_distribution_info:
        print_distribution(all_records, "全量数据")
        print_distribution(records, "采样后数据")

    extract_unique_images(ZIP_PATH, image_name_map, image_dir)
    write_data_json(records, image_dir.parent)
    validate_output(records, image_name_map, image_dir.parent)


if __name__ == "__main__":
    # 转换结果目录；重复运行时只会清除此目录，不会修改原始 data.zip。
    OUTPUT_DIR = "formatted_data"

    # 目标样本总数；None 表示保留官方 Lite 全部 1,919 条。
    TARGET_SAMPLE_COUNT = 200

    # 可选值："sequential" 或 "random"；随机模式不设置随机种子。
    SAMPLING_MODE = "sequential"

    # True：打印 Task、Subtask、Category 分布；False：不打印。
    PRINT_DISTRIBUTION_INFO = True

    convert_dataset(
        output_dir=OUTPUT_DIR,
        target_sample_count=TARGET_SAMPLE_COUNT,
        sampling_mode=SAMPLING_MODE,
        print_distribution_info=PRINT_DISTRIBUTION_INFO,
    )
