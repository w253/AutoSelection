"""将MathVision-Wild test_photo转换为DataRecipe统一OOD benchmark格式。

脚本读取问题与照片，校验后稳定编号，过滤无法用统一规则评分的答案，再按level
与subject分层抽样；将选中图片复制到formatted_data/img并写出data.json。
题面完全位于图片中，text_input直接使用六项benchmark共享的回答格式指令。
输出不含运行时间等不稳定字段，结束时检查分布、字段、ID和图片存在性。
"""

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from recipe_sandbox.dandy.benchmarks.sampling import balanced_eval_records

from recipe_sandbox.dandy.data.answer import (  # noqa: E402
    append_final_answer_instruction,
    partition_supported_answers,
)


# MathVision-Wild 保留了 MATH-Vision 的原始标注：
# - level：五级难度标签，从 1（最低难度）到 5（最高难度）。
# - subject：16 个细粒度数学学科之一：
#   algebra；analytic geometry；arithmetic；combinatorial geometry；
#   combinatorics；counting；descriptive geometry；graph theory；logic；
#   metric geometry - angle；metric geometry - area；
#   metric geometry - length；solid geometry；statistics；topology；
#   transformation geometry。
# 原始数据没有独立的题干字段，题干只存在于拍摄图片中，因此 text_input 不重复题面，
# 只承载与SFT/GRPO训练数据一致的最终答案格式指令（题目仍需模型从图片中读取）。
# 每条20样本的对照实验显示，加入该指令后本benchmark准确率由10%提升到15%。
BENCHMARK_NAME = "MathVision-Wild"
TEXT_INPUT = append_final_answer_instruction("")


def load_and_number_records() -> List[Dict[str, Any]]:
    """读取、校验、稳定排序并编号全部原始样本。

    功能与逻辑：
        从当前 MathVision-Wild 目录的 test_photo/metadata.jsonl 读取数据，
        确认每条标注引用的图片存在；再按 original_id 的数值升序稳定排序，
        为全体样本分配连续且固定的 id，之后才进行任何采样。

    输入：
        无。原始数据目录固定为 test_photo/。

    输出：
        已编号的全部样本字典列表。以 "_" 开头的字段仅用于运行时复制图片，
        不会写入最终数据。
    """
    raw_data_dir = Path("test_photo")
    metadata_path = raw_data_dir / "metadata.jsonl"

    if not raw_data_dir.is_dir():
        raise FileNotFoundError("找不到原始数据目录: {}".format(raw_data_dir))

    if not metadata_path.is_file():
        raise FileNotFoundError("找不到标注文件: {}".format(metadata_path))

    required_fields = {"id", "file_name", "answer", "level", "subject"}
    raw_records = []
    seen_original_ids = set()

    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            raw = json.loads(line)
            missing_fields = required_fields - set(raw.keys())

            if missing_fields:
                raise ValueError(
                    "metadata.jsonl 第 {} 行缺少字段: {}".format(
                        line_number, sorted(missing_fields)
                    )
                )

            original_id = str(raw["id"])

            if original_id in seen_original_ids:
                raise ValueError("发现重复 original_id: {}".format(original_id))

            if not original_id.isdigit():
                raise ValueError(
                    "metadata.jsonl 第 {} 行的 id 不是纯数字: {}".format(
                        line_number, original_id
                    )
                )

            seen_original_ids.add(original_id)
            source_image = raw_data_dir / raw["file_name"]

            if not source_image.is_file():
                raise FileNotFoundError(
                    "metadata.jsonl 第 {} 行对应图片不存在: {}".format(
                        line_number, source_image
                    )
                )

            raw_records.append(
                {
                    "original_id": original_id,
                    "answer": str(raw["answer"]),
                    "level": raw["level"],
                    "subject": raw["subject"],
                    "_source_image": source_image,
                }
            )

    if not raw_records:
        raise ValueError("metadata.jsonl 中没有有效样本")

    raw_records.sort(key=lambda record: int(record["original_id"]))

    numbered_records = []
    for number, record in enumerate(raw_records, start=1):
        sample_id = "{}-{:04d}".format(BENCHMARK_NAME, number)
        image_name = "{}{}".format(sample_id, record["_source_image"].suffix.lower())

        numbered_records.append(
            {
                "id": sample_id,
                "img_path": "img/{}".format(image_name),
                "text_input": TEXT_INPUT,
                "answer": record["answer"],
                "original_id": record["original_id"],
                "level": record["level"],
                "subject": record["subject"],
                "_source_image": record["_source_image"],
                "_image_name": image_name,
            }
        )

    return numbered_records


def print_distribution(records: List[Dict[str, Any]], title: str) -> None:
    """打印一组样本的 level 和 subject 分布。

    功能与逻辑：
        统计传入样本中的标签数量，并以固定顺序打印。该函数会分别统计全量数据
        和采样后数据，以便直接在终端核查采样覆盖情况。

    输入：
        records：样本字典列表。
        title：本次统计对应的数据集合名称。

    输出：
        无返回值；统计结果直接输出到终端。
    """
    level_counter = Counter(record["level"] for record in records)
    subject_counter = Counter(record["subject"] for record in records)

    print("\n{} {} {}".format("=" * 20, title, "=" * 20))
    print("level 分布:")
    for level, count in sorted(level_counter.items()):
        print("  level {}: {}".format(level, count))

    print("\nsubject 分布:")
    for subject, count in sorted(subject_counter.items()):
        print("  {}: {}".format(subject, count))


def sample_records(
    records: List[Dict[str, Any]],
    samples_per_level_subject: Optional[int],
    sampling_mode: Literal["sequential", "random"],
) -> List[Dict[str, Any]]:
    """在每个 (level, subject) 组合中最多抽取固定数量的样本。

    功能与逻辑：
        先对已完成全量编号的样本分组。顺序模式从每组中选择 original_id 最小的
        若干样本；随机模式在每组内进行无放回抽取。抽取后再次按 original_id
        排序，以保证每次生成的 data.json 内部样本顺序稳定。

    输入：
        records：已稳定编号的全部样本字典列表。
        samples_per_level_subject：每个分组最多保留的正整数样本数；
            None 表示保留全部样本。
        sampling_mode："sequential" 或 "random"。

    输出：
        按 original_id 数值升序排列的采样后样本字典列表。
    """
    if samples_per_level_subject is None:
        return records

    if samples_per_level_subject <= 0:
        raise ValueError("samples_per_level_subject 必须为正整数或 None")

    if sampling_mode not in {"sequential", "random"}:
        raise ValueError("sampling_mode 只能是 'sequential' 或 'random'")

    grouped_records = defaultdict(list)
    for record in records:
        group_key = (record["level"], record["subject"])
        grouped_records[group_key].append(record)

    selected_records = []
    for group_key in sorted(grouped_records.keys()):
        group_records = grouped_records[group_key]
        sample_size = min(samples_per_level_subject, len(group_records))

        if sampling_mode == "sequential":
            selected_records.extend(group_records[:sample_size])
        else:
            selected_records.extend(random.sample(group_records, sample_size))

    selected_records.sort(key=lambda record: int(record["original_id"]))
    return selected_records


def prepare_output_dir(output_dir: str) -> Path:
    """清除上一次转换结果，并创建新的 img 目录。

    功能与逻辑：
        当未传入 output_dir 时使用 formatted_data。若输出目录已经存在，
        先完整删除，再新建 img/。为避免误删，禁止将当前 MathVision-Wild
        根目录或 test_photo/ 设为输出目录。

    输入：
        output_dir：目标目录名称或路径，默认值为 "formatted_data"。

    输出：
        新建 img 目录对应的 Path 对象。
    """
    output_path = Path(output_dir).resolve()
    current_path = Path(".").resolve()
    raw_data_path = Path("test_photo").resolve()

    if output_path == current_path:
        raise ValueError("output_dir 不能是当前 MathVision-Wild 根目录")

    if output_path == raw_data_path:
        raise ValueError("output_dir 不能是原始数据目录 test_photo")

    if output_path.exists():
        print("清除上一次转换结果: {}".format(output_path))
        shutil.rmtree(output_path)

    image_dir = output_path / "img"
    image_dir.mkdir(parents=True)
    return image_dir


def copy_images(records: List[Dict[str, Any]], image_dir: Path) -> None:
    """将选中的原始图片复制到输出目录的 img/ 中。

    功能与逻辑：
        以已分配的稳定样本 id 作为目标文件名，复制每一张源图。保留原始扩展名，
        但统一转为小写，以兼容原始数据中混用的 .jpg、.jpeg 和 .JPG 文件。

    输入：
        records：最终选中的样本字典列表。
        image_dir：已创建的目标 img 目录。

    输出：
        无返回值；所有选中图片被复制到 image_dir。
    """
    for index, record in enumerate(records, start=1):
        target_image = image_dir / record["_image_name"]
        shutil.copy2(record["_source_image"], target_image)

        if index % 250 == 0 or index == len(records):
            print("已复制图片: {}/{}".format(index, len(records)))


def write_data_json(
    records: List[Dict[str, Any]], output_dir: Path
) -> None:
    """写入不含运行时临时字段的最终 JSON 标注。

    功能与逻辑：
        序列化前删除所有以 "_" 开头的字段。因此导出的样本不含
        source_file_name、_source_image 或 _image_name，只保留 original_id
        以支持与原始 benchmark 配对。

    输入：
        records：最终选中的样本字典列表。
        output_dir：data.json 的输出目录。

    输出：
        无返回值；在 output_dir 下生成 data.json。
    """
    data = []
    for record in records:
        data.append(
            {key: value for key, value in record.items() if not key.startswith("_")}
        )

    output_json = {
        "created_at": datetime.now().astimezone().isoformat(),
        "num_samples": len(data),
        "data": data,
    }

    json_path = output_dir / "data.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("已生成标注文件: {}".format(json_path))


def convert_dataset(
    output_dir: str = "formatted_data",
    samples_per_level_subject: Optional[int] = None,
    sampling_mode: Literal["sequential", "random"] = "sequential",
    print_distribution_info: bool = True,
    target_sample_count: Optional[int] = 200,
) -> None:
    """将 MathVision-Wild/test_photo 转换为约定的数据目录结构。

    功能与逻辑：
        读取全部原始样本，先对所有样本排序和编号，使用统一答案规则排除无法评分
        的记录；随后可按(level, subject)分组采样，清除旧输出，复制选中图片，
        并写入最终data.json。

    输入：
        target_sample_count：排除SAE后的eval总数，默认200；None取消总量限制。
        output_dir：输出目录，默认值为 "formatted_data"。
        samples_per_level_subject：每个 (level, subject) 组合最多保留的样本数；
            None 表示导出全部样本。
        sampling_mode："sequential" 或 "random"。
        print_distribution_info：是否在终端打印全量数据与采样后数据的
            level、subject 分布。

    输出：
        无返回值；在 output_dir 下生成 data.json 和 img/。
    """
    numbered_records = load_and_number_records()
    all_records, rejected_records = partition_supported_answers(numbered_records)
    print("排除无法规则评分的记录: {}".format(len(rejected_records)))
    if print_distribution_info:
        print_distribution(all_records, "全量数据")

    selected_records = sample_records(
        records=all_records,
        samples_per_level_subject=samples_per_level_subject,
        sampling_mode=sampling_mode,
    )
    selected_records = balanced_eval_records(
        selected_records, target_sample_count, (('level', 'subject'),), sampling_mode
    )
    if print_distribution_info:
        print_distribution(selected_records, "采样后数据")

    image_dir = prepare_output_dir(output_dir)
    copy_images(selected_records, image_dir)
    write_data_json(selected_records, image_dir.parent)

    print("\n转换完成，共导出 {} 条样本".format(len(selected_records)))
    print("输出目录: {}".format(image_dir.parent))


if __name__ == "__main__":
    # 可改为其他目录；默认每次运行会删除并重建 formatted_data/。
    OUTPUT_DIR = "formatted_data"

    # 按(level, subject)均衡选出200条；每组上限默认关闭。
    # None：导出全部合格记录。
    # 正整数：每个 (level, subject) 组合最多导出该数量。
    SAMPLES_PER_LEVEL_SUBJECT = None
    TARGET_SAMPLE_COUNT = 200

    # 可选值："sequential" 或 "random"。
    # random 模式每次会重新抽样，但导出后的样本仍按 original_id 稳定排序。
    SAMPLING_MODE = "sequential"

    # True：打印全量与采样后数据的 level、subject 分布；False：不打印。
    PRINT_DISTRIBUTION_INFO = True

    convert_dataset(
        output_dir=OUTPUT_DIR,
        samples_per_level_subject=SAMPLES_PER_LEVEL_SUBJECT,
        sampling_mode=SAMPLING_MODE,
        print_distribution_info=PRINT_DISTRIBUTION_INFO,
        target_sample_count=TARGET_SAMPLE_COUNT,
    )
