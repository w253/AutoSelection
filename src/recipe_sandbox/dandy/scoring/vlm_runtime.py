"""执行Qwen2.5-VL的SAE特征、IFD和Varentropy真实模型前向。

本模块只实现当前SFT/GRPO配置实际使用的Qwen2.5-VL运行适配，不读取外部
``AutoSelection/``代码。SAE路径在匹配产物的固定基础VLM上提取指定语言层全部
非padding激活并调用encoder-only SAE；语言指标路径在阶段实际起始checkpoint上
计算图文条件回答与answer-only回答的assistant正文token统计。两条路径可顺序加载，
避免把GRPO的SFT checkpoint误当作SAE固定基座，也避免两个VLM同时驻留NPU。

assistant mask不依赖Transformers当前存在多模态错位风险的自动mask，而是从Qwen
模板处理后的真实input_ids中识别``<|im_start|>assistant\n``到``<|im_end|>``之间的
正文token。模板结构、回答段数、图片token或目标层不匹配时立即失败，不采用猜测
边界或全token回退。本模块只在实例化运行类时导入Torch、Transformers和sparsify。
"""

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .records import SparseFeatureRow
from .language_metrics import (
    assistant_nll_from_logits,
    assistant_varentropy_from_logits,
)
from .metric_inputs import (
    MetricConversation,
    answer_only_conversation,
    qwen_dialog,
)
from .records import MeanStatistic
from .sae_encoding import encode_sample_sae_feature


class SequenceTooLongError(ValueError):
    """携带实际processor长度；仅此异常允许缓存跳过训练池样本。"""

    def __init__(self, sample_id: str, length: int, limit: int):
        """保存超限输入的身份、实测token数和上限，不持有模型或输入张量。"""
        self.sample_id, self.length, self.limit = sample_id, length, limit
        super().__init__(f"样本{sample_id}序列长度{length}超过max_length {limit}")


class _StopAfterHook(Exception):
    """在SAE目标语言层输出已经捕获后终止剩余VLM前向。"""


@dataclass(frozen=True)
class LanguageMetricResult:
    """保存一条样本的IFD两项NLL与条件Varentropy充分统计量。"""

    conditional_nll: MeanStatistic
    answer_only_nll: MeanStatistic
    varentropy: MeanStatistic


def _sum_statistics(values: Sequence[MeanStatistic]) -> MeanStatistic:
    """合并多段回答的同类充分统计量。

    参数：
        values：至少一个MeanStatistic；每项已经通过有限性和正计数校验。

    返回：
        总和与计数分别相加的新MeanStatistic。
    """
    if not values:
        raise ValueError("至少需要一段语言统计量")
    return MeanStatistic(
        total=sum(value.total for value in values),
        count=sum(value.count for value in values),
    )


def _find_subsequence(values: Sequence[int], needle: Sequence[int], start: int) -> int:
    """从指定位置查找整数子序列的首个精确匹配。

    参数：
        values：被搜索的token ID序列。
        needle：非空目标token ID序列。
        start：非负起始位置。

    返回：
        首个匹配的零基位置；不存在时返回-1。
    """
    if not needle:
        raise ValueError("token子序列不能为空")
    for position in range(start, len(values) - len(needle) + 1):
        if list(values[position : position + len(needle)]) == list(needle):
            return position
    return -1


def _hf_attention_implementation(value: str) -> str:
    """把阶段YAML的ms-swift注意力名称转换成Transformers名称。

    参数：
        value：阶段YAML或SAE产物记录的注意力实现名称。

    返回：
        Transformers ``from_pretrained``接受的名称；当前支持eager、sdpa和两种
        FlashAttention 2拼写，未知值直接报错，避免静默改变计算后端。
    """
    mapping = {
        "eager": "eager",
        "sdpa": "sdpa",
        "flash_attn": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
    }
    if value not in mapping:
        raise ValueError("不支持的Qwen注意力实现: {}".format(value))
    return mapping[value]


def assistant_content_spans(
    input_ids: Sequence[int],
    *,
    assistant_start_ids: Sequence[int],
    message_end_ids: Sequence[int],
    expected_segments: int,
) -> tuple[tuple[int, int], ...]:
    """从Qwen模板token序列解析全部assistant正文半开区间。

    参数：
        input_ids：processor展开视觉token后的单样本token ID序列。
        assistant_start_ids：``<|im_start|>assistant\n``的token ID序列。
        message_end_ids：``<|im_end|>``的token ID序列。
        expected_segments：规范消息中预期的assistant段数。

    返回：
        ``(content_start, content_end)``元组序列；角色头、消息结束符和其他角色
        token均不在区间内。模板边界缺失、正文为空或段数不一致时抛出ValueError。
    """
    if expected_segments <= 0:
        raise ValueError("assistant span至少需要一个回答段")
    ids = [int(value) for value in input_ids]
    starts = tuple(int(value) for value in assistant_start_ids)
    ends = tuple(int(value) for value in message_end_ids)
    if not starts or not ends:
        raise ValueError("assistant模板边界token不能为空")
    spans = []
    cursor = 0
    while True:
        start = _find_subsequence(ids, starts, cursor)
        if start < 0:
            break
        content_start = start + len(starts)
        end = _find_subsequence(ids, ends, content_start)
        if end < 0:
            raise ValueError("Qwen assistant消息缺少<|im_end|>边界")
        if end <= content_start:
            raise ValueError("Qwen assistant正文没有token")
        spans.append((content_start, end))
        cursor = end + len(ends)
    if len(spans) != expected_segments:
        raise ValueError(
            "Qwen assistant模板段数{}与回答数{}不一致".format(
                len(spans), expected_segments
            )
        )
    return tuple(spans)


def prepare_qwen_inputs(processor, conversation: MetricConversation) -> Dict[str, Any]:
    """在CPU使用正式图文模板及processor构造输入，供长度预检和模型前向共用。

    不加载VLM或移动张量到设备；返回包含真实视觉token的batch。图片对象及时关闭。
    不截断、不过滤，不捕获解码/格式错误，由调用方决定长度超限策略。
    """
    from PIL import Image

    dialog = qwen_dialog(conversation)
    text = processor.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=False,
    )
    opened = []
    try:
        for path in conversation.images:
            with Image.open(path) as source:
                opened.append(source.convert("RGB"))
        batch = dict(
            processor(
                text=[text],
                images=opened or None,
                return_tensors="pt",
                padding=False,
            )
        )
    finally:
        for image in opened:
            image.close()
    return batch


class QwenMetricModel:
    """封装一个只读Qwen2.5-VL及其processor和可选语言层钩子。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str,
        dtype: str,
        max_pixels: int,
        max_length: int,
        attn_implementation: str,
        processor_use_fast: bool = True,
        expected_processor_class: str | None = None,
        hookpoint: str | None = None,
    ) -> None:
        """加载冻结Qwen2.5-VL并核对processor、序列边界与可选目标层。

        参数：
            model_path：本地Qwen2.5-VL基础模型或阶段checkpoint目录。
            device：Torch设备字符串；NPU设备会显式导入torch_npu并设置当前设备。
            dtype：只允许bfloat16、float16或float32。
            max_pixels：processor单图正整数像素上限。
            max_length：允许的完整图文序列正整数长度上限，超长样本整体报错。
            attn_implementation：传给Transformers模型加载器的注意力实现。
            processor_use_fast：是否要求fast processor。
            expected_processor_class：可选的SAE产物processor类名绑定。
            hookpoint：可选的完整语言层路径；提供时解析并保存该层。

        返回：
            无；实例保存模型、processor、设备和运行元信息，并占用相应设备内存。
        """
        import torch

        if str(device).startswith("npu"):
            importlib.import_module("torch_npu")  # 注册Ascend设备后端。
        torch_device = torch.device(device)
        if torch_device.type == "npu":
            torch.npu.set_device(torch_device)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Qwen指标dtype必须是bfloat16、float16或float32")
        if (
            isinstance(max_pixels, bool)
            or not isinstance(max_pixels, int)
            or max_pixels <= 0
            or isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length <= 0
        ):
            raise ValueError("Qwen指标max_pixels和max_length必须是正整数")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_path = Path(model_path).resolve()
        self.device = torch_device
        self.dtype_name = dtype
        self.max_pixels = max_pixels
        self.max_length = max_length
        resolved_attention = _hf_attention_implementation(attn_implementation)
        self.attn_implementation = resolved_attention
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            max_pixels=max_pixels,
            use_fast=processor_use_fast,
        )
        processor_class = type(self.processor).__name__
        if (
            expected_processor_class is not None
            and processor_class != expected_processor_class
        ):
            raise ValueError(
                "SAE产物要求processor类{}，实际为{}".format(
                    expected_processor_class, processor_class
                )
            )
        torch_dtype = getattr(torch, dtype)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            dtype=torch_dtype,
            attn_implementation=resolved_attention,
            low_cpu_mem_usage=True,
        )
        model_type = getattr(self.model.config, "model_type", None)
        if model_type not in {"qwen2_5_vl", "qwen2_5_vl_text"}:
            raise ValueError("当前指标运行层只支持Qwen2.5-VL")
        self.model.requires_grad_(False).eval().to(self.device)
        self.model.config.use_cache = False
        self.image_token_id = int(self.model.config.image_token_id)
        self.hookpoint = hookpoint
        self.layer = None
        if hookpoint is not None:
            try:
                self.layer = self.model.get_submodule(hookpoint)
            except AttributeError as error:
                raise ValueError(
                    "SAE hookpoint在当前基础VLM中不存在: {}".format(hookpoint)
                ) from error
        self._assistant_start_ids = tuple(
            self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
        )
        self._message_end_ids = tuple(
            self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        )
        if not self._assistant_start_ids or not self._message_end_ids:
            raise ValueError("Qwen tokenizer无法编码assistant模板边界")
        self.runtime_info = {
            "model_path": str(self.model_path),
            "model_class": type(self.model).__name__,
            "model_type": model_type,
            "processor_class": processor_class,
            "processor_use_fast": bool(processor_use_fast),
            "device": str(self.device),
            "dtype": dtype,
            "max_pixels": max_pixels,
            "max_length": max_length,
            "configured_attn_implementation": attn_implementation,
            "attn_implementation": resolved_attention,
            "assistant_mask": "qwen_template_content_boundaries_v1",
            "hookpoint": hookpoint,
        }

    def close(self) -> None:
        """释放模型和processor引用并清理当前加速设备缓存。

        参数：
            无。

        返回：
            无；实例关闭后不可再次执行前向。NPU/CUDA存在时调用对应empty_cache，
            CPU运行只删除对象引用。
        """
        torch = self.torch
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        self.layer = None
        if model is not None:
            del self.model
        if processor is not None:
            del self.processor
        # 局部变量同样持有对象引用；必须先删除并回收，再清理设备分配器缓存。
        # 否则模型会在empty_cache返回后才析构，父进程继续占用NPU保留显存，
        # 随后启动的colocate vLLM子进程可能无法分配任何KV cache block。
        del model
        del processor
        gc.collect()
        if self.device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
        elif self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_batch(self, conversation: MetricConversation) -> Dict[str, Any]:
        """应用Qwen原生模板、读取全部图片并构造单样本模型输入。

        参数：
            conversation：图片占位符已经严格对齐的规范对话。

        返回：
            位于目标设备的processor张量字典；序列超长、图片token缺失或图片解码
            失败时抛出异常，不截断文本或视觉位置。
        """
        batch = prepare_qwen_inputs(self.processor, conversation)
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if (
            input_ids is None
            or attention_mask is None
            or input_ids.ndim != 2
            or tuple(input_ids.shape) != tuple(attention_mask.shape)
            or input_ids.shape[0] != 1
        ):
            raise ValueError("Qwen processor没有返回单样本input_ids/attention_mask")
        if input_ids.shape[1] > self.max_length:
            raise SequenceTooLongError(
                conversation.sample_id, int(input_ids.shape[1]), self.max_length
            )
        visual_tokens = int((input_ids[0] == self.image_token_id).sum().item())
        if conversation.images and visual_tokens <= 0:
            raise ValueError("图文样本没有产生Qwen视觉token")
        if not conversation.images and visual_tokens:
            raise ValueError("纯文本answer-only输入意外产生视觉token")
        result = {}
        for key, value in batch.items():
            if not isinstance(value, self.torch.Tensor):
                continue
            target_dtype = (
                getattr(self.torch, self.dtype_name)
                if key == "pixel_values"
                else value.dtype
            )
            result[key] = value.to(device=self.device, dtype=target_dtype)
        return result

    def _assistant_mask(
        self,
        input_ids,
        *,
        expected_segments: int,
    ):
        """在真实processor token序列中定位全部assistant正文。

        参数：
            input_ids：形状``[1, sequence]``的Qwen token ID张量。
            expected_segments：规范对话中assistant消息数量。

        返回：
            与input_ids同形状的bool张量，只标记角色头之后、``im_end``之前的正文；
            模板边界缺失、空回答或段数不一致时抛出ValueError。
        """
        ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        mask = self.torch.zeros_like(input_ids, dtype=self.torch.bool)
        spans = assistant_content_spans(
            ids,
            assistant_start_ids=self._assistant_start_ids,
            message_end_ids=self._message_end_ids,
            expected_segments=expected_segments,
        )
        for content_start, end in spans:
            mask[0, content_start:end] = True
        return mask

    def extract_activations(self, conversation: MetricConversation):
        """提取目标语言层全部非padding token的FP32 CPU激活。

        参数：
            conversation：训练池完整对话或不含答案的benchmark问题对话。

        返回：
            形状``[nonpadding_tokens, hidden_size]``的FP32 CPU张量；目标层未配置、
            捕获形状不匹配、视觉token缺失或非有限值时报错。
        """
        if self.layer is None or self.hookpoint is None:
            raise ValueError("当前Qwen运行实例没有配置SAE hookpoint")
        batch = self._prepare_batch(conversation)
        captured = None

        def capture(module, inputs, output) -> None:
            """保存目标层输出并用局部异常终止后续模型计算。"""
            nonlocal captured
            captured = (
                output[0] if isinstance(output, (tuple, list)) else output
            ).detach()
            raise _StopAfterHook()

        handle = self.layer.register_forward_hook(capture)
        try:
            with self.torch.no_grad():
                try:
                    self.model(**batch, use_cache=False)
                except _StopAfterHook:
                    pass
        finally:
            handle.remove()
        if (
            captured is None
            or captured.ndim != 3
            or tuple(captured.shape[:2]) != tuple(batch["input_ids"].shape)
        ):
            raise ValueError("SAE目标层激活与Qwen token位置不一致")
        attention_mask = batch["attention_mask"][0].bool()
        activations = captured[0][attention_mask].float().cpu()
        if not len(activations) or not bool(self.torch.isfinite(activations).all().item()):
            raise ValueError("SAE目标层激活为空或包含非有限值")
        return activations

    def language_statistics(
        self,
        conversation: MetricConversation,
        *,
        varentropy_token_chunk_size: int = 32,
    ) -> LanguageMetricResult:
        """计算图文条件回答、answer-only NLL与条件Varentropy。

        逻辑：完整对话只前向一次并同时复用logits计算条件NLL和Varentropy；随后每个
        assistant正文各构造一条无图片、无instruction但保留assistant模板头的对话，
        分别前向后合并answer-only充分统计量。所有指标只选择回答正文token，不把
        system/user、视觉token、角色控制符或``im_end``计入分母。

        参数：
            conversation：至少含一个assistant回答的训练池规范对话。
            varentropy_token_chunk_size：完整词表Varentropy的正整数token分块大小。

        返回：
            LanguageMetricResult；模板边界、模型logits或统计量异常时直接报错。
        """
        if not conversation.answers:
            raise ValueError("语言指标要求至少一个assistant回答")
        batch = self._prepare_batch(conversation)
        assistant_mask = self._assistant_mask(
            batch["input_ids"], expected_segments=len(conversation.answers)
        )
        with self.torch.no_grad():
            outputs = self.model(**batch, use_cache=False)
        logits = outputs.logits[0, :-1]
        target_ids = batch["input_ids"][0, 1:]
        target_mask = assistant_mask[0, 1:]
        conditional = assistant_nll_from_logits(logits, target_ids, target_mask)
        varentropy = assistant_varentropy_from_logits(
            logits,
            target_ids,
            target_mask,
            token_chunk_size=varentropy_token_chunk_size,
        )
        answer_statistics = []
        for position, answer in enumerate(conversation.answers):
            answer_conversation = answer_only_conversation(answer, position)
            answer_batch = self._prepare_batch(answer_conversation)
            answer_mask = self._assistant_mask(
                answer_batch["input_ids"], expected_segments=1
            )
            with self.torch.no_grad():
                answer_outputs = self.model(**answer_batch, use_cache=False)
            answer_statistics.append(
                assistant_nll_from_logits(
                    answer_outputs.logits[0, :-1],
                    answer_batch["input_ids"][0, 1:],
                    answer_mask[0, 1:],
                )
            )
            del answer_outputs
        del outputs
        return LanguageMetricResult(
            conditional_nll=conditional,
            answer_only_nll=_sum_statistics(answer_statistics),
            varentropy=varentropy,
        )


class QwenSaeFeatureRuntime:
    """组合固定基础Qwen、匹配SAE编码器与样本级Top-192导出。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        artifact,
        device: str,
        dtype: str,
        token_batch_size: int,
    ) -> None:
        """按SAE产物绑定信息加载基础VLM和encoder-only SAE。

        参数：
            model_path：基础VLM目录；不比较复制前后的文件标记。
            artifact：已由``load_sae_artifact_contract``校验的产物契约。
            device：Torch设备字符串。
            dtype：基础VLM前向精度，通常来自阶段YAML。
            token_batch_size：SAE编码时的正整数token分块大小。

        返回：
            无；实例持有基础VLM与FP32 SAE并占用设备内存。产物adapter、输入策略、
            hookpoint不存在或特征维度不匹配时抛出异常。
        """
        from .sae_loader import load_encoder

        metadata = artifact.metadata
        if metadata.get("adapter") != "qwen2_5_vl":
            raise ValueError("当前指标运行层只实现Qwen2.5-VL SAE产物")
        policy = metadata["input_policy"]
        required_policy = {
            "include_answers",
            "max_pixels",
            "max_dynamic_patch",
            "max_length",
            "attn_implementation",
            "processor_use_fast",
            "processor_class",
        }
        if set(policy) != required_policy:
            raise ValueError("SAE产物input_policy字段与当前固定契约不一致")
        if policy["include_answers"] is not True:
            raise ValueError("AutoSelection SAE产物必须使用include_answers=true")
        if (
            isinstance(token_batch_size, bool)
            or not isinstance(token_batch_size, int)
            or token_batch_size <= 0
        ):
            raise ValueError("SAE token_batch_size必须是正整数")
        self.model_runtime = QwenMetricModel(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_pixels=int(policy["max_pixels"]),
            max_length=int(policy["max_length"]),
            attn_implementation=str(policy["attn_implementation"]),
            processor_use_fast=bool(policy["processor_use_fast"]),
            expected_processor_class=str(policy["processor_class"]),
            hookpoint=str(metadata["hookpoint"]),
        )
        self.encoder = load_encoder(
            artifact.directory,
            str(metadata["hookpoint"]),
            policy,
            device=device,
        )
        self.artifact = artifact
        self.token_batch_size = token_batch_size
        self.runtime_info = {
            **self.model_runtime.runtime_info,
            "artifact_path": str(artifact.directory),
            "artifact_type": metadata["artifact_type"],
            "artifact_dtype": metadata["dtype"],
            "pooling": metadata["pooling"],
            "sample_top_k": 192,
            "sae_dimension": int(self.encoder.num_latents),
            "token_batch_size": token_batch_size,
        }

    def close(self) -> None:
        """释放SAE编码器与基础VLM设备内存。

        参数：
            无。

        返回：
            无；实例关闭后不可再次编码样本。
        """
        if hasattr(self, "encoder"):
            del self.encoder
        self.model_runtime.close()

    def encode(self, conversation: MetricConversation) -> SparseFeatureRow:
        """提取一条图文对话的目标层激活并编码为Top-192 SAE特征。

        参数：
            conversation：训练池完整对话或benchmark问题对话。

        返回：
            以conversation.sample_id为键的SparseFeatureRow。
        """
        activations = self.model_runtime.extract_activations(conversation)
        return encode_sample_sae_feature(
            self.encoder,
            activations,
            sample_id=conversation.sample_id,
            token_batch_size=self.token_batch_size,
        )


class QwenLanguageMetricRuntime:
    """在阶段实际起始checkpoint上计算IFD与Varentropy。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        stage_config: Mapping[str, Any],
        device: str,
        varentropy_token_chunk_size: int,
    ) -> None:
        """按阶段训练YAML加载语言指标Qwen模型。

        参数：
            model_path：SFT配置模型或按运行规则选出的GRPO实际起始checkpoint。
            stage_config：对应阶段完整YAML字典，提供dtype、max_pixels、max_length和
                attn_impl，其他训练字段不被本运行层覆盖。
            device：Torch设备字符串。
            varentropy_token_chunk_size：完整词表Varentropy的正整数token分块大小。

        返回：
            无；实例持有只读模型。缺少字段或非Qwen模型时抛出异常。
        """
        required = {"torch_dtype", "max_pixels", "max_length", "attn_impl"}
        missing = sorted(required - set(stage_config))
        if missing:
            raise ValueError("阶段配置缺少语言指标字段: {}".format(missing))
        if (
            isinstance(varentropy_token_chunk_size, bool)
            or not isinstance(varentropy_token_chunk_size, int)
            or varentropy_token_chunk_size <= 0
        ):
            raise ValueError("Varentropy token分块大小必须是正整数")
        effective_max_length = int(stage_config["max_length"])
        if "max_completion_length" in stage_config:
            completion_length = int(stage_config["max_completion_length"])
            if completion_length <= 0:
                raise ValueError("GRPO max_completion_length必须是正整数")
            effective_max_length += completion_length
        self.model_runtime = QwenMetricModel(
            model_path=model_path,
            device=device,
            dtype=str(stage_config["torch_dtype"]),
            max_pixels=int(stage_config["max_pixels"]),
            max_length=effective_max_length,
            attn_implementation=str(stage_config["attn_impl"]),
            processor_use_fast=True,
        )
        self.runtime_info = dict(self.model_runtime.runtime_info)
        self.varentropy_token_chunk_size = varentropy_token_chunk_size
        self.runtime_info["varentropy_token_chunk_size"] = varentropy_token_chunk_size
        self.runtime_info["configured_prompt_or_sequence_max_length"] = int(
            stage_config["max_length"]
        )
        self.runtime_info["configured_max_completion_length"] = stage_config.get(
            "max_completion_length"
        )

    def close(self) -> None:
        """释放阶段语言指标VLM与processor设备内存。

        参数：
            无。

        返回：
            无；实例关闭后不可再次计算语言指标。
        """
        self.model_runtime.close()

    def compute(self, conversation: MetricConversation) -> LanguageMetricResult:
        """计算一条训练池对话的IFD和Varentropy充分统计量。

        参数：
            conversation：至少包含一个回答的训练池规范对话。

        返回：
            条件NLL、answer-only NLL与条件Varentropy三组充分统计量。
        """
        return self.model_runtime.language_statistics(
            conversation,
            varentropy_token_chunk_size=self.varentropy_token_chunk_size,
        )
