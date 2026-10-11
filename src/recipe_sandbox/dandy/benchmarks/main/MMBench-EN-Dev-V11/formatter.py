"""转换Ryoo72/MMBench-EN-Dev-V11为固定200题的单次评测子集及50题SAE。

仅读取all/dev-*.parquet，不重复读取六个能力子集。按官方index % 1000000
识别CircularEval题组，只保留index等于题组ID的原始版本；先稳定编号，再排除
固定SAE，按L2能力和L3类别均衡选题。输出共享回答指令、图片及互斥样本清单。
这不是官方CircularEval分数。原始文件不修改，重复执行只重建本目录formatted_data。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path.cwd()  # 在外部 benchmark 数据目录运行，不把数据写入源码目录
sys.path.insert(0, str(Path(__file__).resolve().parents[6] / 'src'))
from recipe_sandbox.dandy.data.answer import append_final_answer_instruction, partition_supported_answers
from recipe_sandbox.dandy.benchmarks.sampling import balanced_eval_records

BENCHMARK_NAME = 'MMBench-EN-Dev-V11'
SOURCE_URL = 'https://huggingface.co/datasets/Ryoo72/MMBench-EN-Dev-V11'
LEVELS = (('l2_category',), ('category',))
L2_CATEGORIES = {
    'coarse_perception', 'finegrained_perception (instance-level)',
    'finegrained_perception (cross-instance)', 'attribute_reasoning',
    'relation_reasoning', 'logic_reasoning',
}
SAE_REFERENCE_COUNT = 200  # 固定参考集，不随本次eval目标调整。


def text_value(value: Any, label: str, required: bool = True) -> str:
    """规范化Parquet文本；可选字段允许None/NaN/空串，必填缺失或非字符串报错。"""
    if value is None or isinstance(value, float) and math.isnan(value):
        value = ''
    if not isinstance(value, str) or required and not value.strip():
        raise ValueError(f'{label}必须为有效文本，收到{value!r}')
    return value.strip()


def load_records(data_dir: Path) -> tuple[list[dict], dict]:
    """读取all分片并返回原始版本的统一记录及行数审计，不写文件。

    校验index唯一、dev划分、基准版本存在、题组内题干/能力一致；丢弃轮换版本，
    不将它们作为独立题分给SAE。图片字节暂存在私有字段，仅入选图片会落盘。
    选项允许2至4个，但标签必须连续，答案必须落在实际非空选项内。
    """
    paths = sorted(data_dir.glob('dev-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'找不到{data_dir}/dev-*.parquet；请下载all配置')
    rows: dict[int, dict] = {}
    required = {'index', 'question', 'hint', 'A', 'B', 'C', 'D', 'answer',
                'category', 'l2-category', 'image', 'split'}
    for path in paths:
        source = pq.ParquetFile(path)
        missing = required - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f'{path}缺少字段：{sorted(missing)}')
        for batch in source.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                index = row['index']
                if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in rows:
                    raise ValueError(f'非法或重复index：{index}')
                if row['split'] != 'dev':
                    raise ValueError(f'非dev记录：{index}')
                rows[index] = row
    for index, row in rows.items():
        base = rows.get(index % 1_000_000)
        if base is None:
            raise ValueError(f'轮换记录{index}缺少原始版本')
        if any(row[key] != base[key] for key in ('question', 'category', 'l2-category')):
            raise ValueError(f'题组{index % 1_000_000}的文本或类别不一致')
    records = []
    for index in sorted(i for i in rows if i < 1_000_000):
        row = rows[index]
        options = {key: text_value(row[key], f'{index}.{key}', False) for key in 'ABCD'}
        options = {key: value for key, value in options.items() if value}
        if len(options) < 2 or list(options) != list('ABCD'[:len(options)]):
            raise ValueError(f'{index}选项缺失或不连续')
        answer = text_value(row['answer'], f'{index}.answer').upper()
        if answer not in options:
            raise ValueError(f'{index}答案不在有效选项中：{answer}')
        l2 = text_value(row['l2-category'], f'{index}.l2-category')
        if l2 not in L2_CATEGORIES:
            raise ValueError(f'未知L2能力：{l2}')
        question = text_value(row['question'], f'{index}.question')
        hint = text_value(row['hint'], f'{index}.hint', False)
        prompt = '\n'.join(([hint] if hint else []) + [question] +
                           [f'{key}. {value}' for key, value in options.items()])
        image = row['image']
        if not isinstance(image, dict) or not image.get('bytes'):
            raise ValueError(f'{index}缺少嵌入图片bytes；此formatter不接受未解析的图片引用')
        records.append({'id': f'{BENCHMARK_NAME}_{index:06d}', 'original_id': str(index),
                        'text_input': append_final_answer_instruction(prompt), 'answer': answer,
                        'category': text_value(row['category'], f'{index}.category'),
                        'l2_category': l2, '_image_bytes': image['bytes']})
    supported, rejected = partition_supported_answers(records)
    return supported, {'source_rows': len(rows), 'unique_questions': len(records),
                       'discarded_rotations': len(rows) - len(records),
                       'unsupported_answers': len(rejected), 'parquet_files': [p.name for p in paths]}


def select_samples(records: list[dict], target: int | None, sae_count: int) -> tuple[list[dict], list[dict]]:
    """固定参考200题之外按ID哈希选SAE，再对其补集两级均衡选eval；返回有序两组。

    SAE选择与target无关。容量不足时报错，不重复补样；target=None保留全部非SAE题。
    """
    if isinstance(sae_count, bool) or not isinstance(sae_count, int) or sae_count < 1:
        raise ValueError('sae_count必须为正整数')
    reference = balanced_eval_records(records, SAE_REFERENCE_COUNT, LEVELS)
    reference_ids = {row['id'] for row in reference}
    candidates = [row for row in records if row['id'] not in reference_ids]
    if len(candidates) < sae_count:
        raise ValueError('参考eval之外没有足够SAE样本')
    chosen = sorted(candidates, key=lambda row: (hashlib.sha256(row['id'].encode()).digest(), row['id']))[:sae_count]
    sae_ids = {row['id'] for row in chosen}
    sae = [row for row in records if row['id'] in sae_ids]
    evaluation = balanced_eval_records([row for row in records if row['id'] not in sae_ids], target, LEVELS)
    return evaluation, sae


def image_extension(blob: bytes) -> str:
    """用Pillow检查图片字节并返回真实格式扩展名；不重编码，不改变分辨率。"""
    with Image.open(io.BytesIO(blob)) as image:
        extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif',
                     'BMP': '.bmp', 'TIFF': '.tiff'}.get(image.format)
        image.verify()
    if extension is None:
        raise ValueError('不支持的图片格式')
    return extension


def convert_dataset(target_sample_count: int | None = 200, sae_sample_count: int = 50) -> None:
    """从脚本所在目录的all读取数据，校验后重建固定formatted_data输出。

    先验证所选图片及划分，才清除旧输出；拒绝符号链接及非目录输出，防止越界。
    写出data/sae_data及计数、来源、单次评测协议和审计信息，再回读核对ID和图片。
    """
    records, audit = load_records(ROOT / 'all')
    evaluation, sae = select_samples(records, target_sample_count, sae_sample_count)
    selected = evaluation + sae
    for row in selected:
        row['img_path'] = 'img/' + row['id'] + image_extension(row['_image_bytes'])
    output = ROOT / 'formatted_data'
    if output.is_symlink() or output.exists() and not output.is_dir():
        raise ValueError(f'不安全的输出路径：{output}')
    if output.resolve().parent != ROOT.resolve():
        raise ValueError('输出路径越界')
    if output.exists():
        shutil.rmtree(output)
    (output / 'img').mkdir(parents=True)
    for row in selected:
        (output / row['img_path']).write_bytes(row['_image_bytes'])
    payload = {'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
               'source': SOURCE_URL, 'evaluation_protocol': 'single_pass_original_options',
               'audit': audit, 'num_samples': len(evaluation), 'num_sae_samples': len(sae)}
    for key, group in (('data', evaluation), ('sae_data', sae)):
        payload[key] = [{k: v for k, v in row.items() if not k.startswith('_')} for row in group]
    path = output / 'data.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    saved = json.loads(path.read_text(encoding='utf-8'))
    combined = saved['data'] + saved['sae_data']
    if len({row['id'] for row in combined}) != len(combined):
        raise ValueError('eval/SAE样本ID重复')
    if saved['num_samples'] != len(saved['data']) or saved['num_sae_samples'] != len(saved['sae_data']):
        raise ValueError('输出计数不一致')
    expected = {output / row['img_path'] for row in combined}
    if expected != set((output / 'img').iterdir()):
        raise ValueError('输出图片与JSON引用不一致')
    print(json.dumps(audit, ensure_ascii=False))
    print('eval L2分布:', dict(Counter(row['l2_category'] for row in evaluation)))
    print('eval L3分布:', dict(Counter(row['category'] for row in evaluation)))
    print(f'校验通过：eval={len(evaluation)}，SAE={len(sae)}，输出={output}')


if __name__ == '__main__':
    TARGET_SAMPLE_COUNT = 200
    SAE_SAMPLE_COUNT = 50
    convert_dataset(TARGET_SAMPLE_COUNT, SAE_SAMPLE_COUNT)
