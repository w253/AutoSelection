"""MS-SWIFT 4.0.1 final-resample fix, loaded only for dynamic GRPO.

No site-packages edits. DR_RESAMPLE_AUDIT_DIR enables opt-in distributed
sample/reward alignment checks and rank-0 JSONL evidence during a short test.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap


def patched_source(source: str) -> str:
    """只替换已核对的 4.0.1 重采样边界；源码结构变化时拒绝套用。"""
    source = textwrap.dedent(source)
    loop = "while resample_count < self.max_resample_times:"
    next_batch = "        inputs = next(self.dynamic_resample_iterator)"
    if source.count(loop) != 1 or source.count(next_batch) != 1:
        raise RuntimeError("GRPO补采补丁与安装源码不匹配；请核对版本，不能静默跳过")
    source = source.replace(loop, "while True:", 1)
    source = source.replace(next_batch,
        "        if resample_count >= self.max_resample_times:\n"
        "            break\n\n" + next_batch, 1)
    return source


def install():
    """由 SWIFT 显式加载时安装原有 GRPO 补丁，不修改 site-packages。"""
    from importlib.metadata import version
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    current = GRPOTrainer._dynamic_sampling
    if getattr(current, "_datarecipe_final_resample_fix", False):
        return
    if version("ms-swift") != "4.0.1":
        raise RuntimeError("GRPO补采补丁仅验证ms-swift 4.0.1；升级后须重新核对")
    original = inspect.unwrap(current)
    source = inspect.getsource(original)
    tree = ast.parse(patched_source(source))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<datarecipe-final-resample-fix>", "exec"), namespace)
    fixed = namespace[original.__name__]

    @functools.wraps(current)
    def run(self, inputs, rewards_per_func):
        audit_dir = os.environ.get("DR_RESAMPLE_AUDIT_DIR")
        if not audit_dir:
            return fixed(self, inputs, rewards_per_func)
        import torch
        from accelerate.utils import gather_object

        def snapshot(items, rewards):
            signatures = [hashlib.sha256(json.dumps(
                {k: item.get(k) for k in ("id", "prompt_id", "request_id", "messages")},
                sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
                for item in items]
            return dict(signatures=gather_object(signatures),
                        rewards=rewards.detach().float().cpu().tolist(),
                        valid=(self.compute_std(items, rewards)>0).detach().cpu().tolist())

        attempts = [snapshot(inputs, rewards_per_func)]
        score = self._score_completions
        had_instance_score = "_score_completions" in self.__dict__
        old_instance_score = self.__dict__.get("_score_completions")
        def audited_score(items, *args, **kwargs):
            rewards = score(items, *args, **kwargs)
            attempts.append(snapshot(items, rewards))
            return rewards
        self._score_completions = audited_score
        try:
            result = fixed(self, inputs, rewards_per_func)
        finally:
            if had_instance_score:
                self._score_completions = old_instance_score
            else:
                del self._score_completions
        final = snapshot(*result)
        target = self.args.generation_batch_size
        selected = [(sig, reward) for a in attempts
                    for sig, reward, valid in zip(a["signatures"], a["rewards"], a["valid"])
                    if valid][:target]
        enough = len(selected) == target
        expected_sig = [v[0] for v in selected] if enough else attempts[0]["signatures"]
        expected_rewards = [v[1] for v in selected] if enough else attempts[0]["rewards"]
        aligned = (final["signatures"] == expected_sig and
                   len(final["rewards"]) == len(expected_rewards) and
                   torch.allclose(torch.tensor(final["rewards"]),
                                  torch.tensor(expected_rewards), rtol=0, atol=0, equal_nan=True))
        bounded = len(attempts)-1 <= self.max_resample_times
        if self.accelerator.is_main_process:
            dest = Path(audit_dir)
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/"resample-audit.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(dict(step=self.state.global_step,
                    attempts=attempts, final=final, enough_valid=enough,
                    resample_calls=len(attempts)-1, aligned=bool(aligned),
                    within_limit=bounded), ensure_ascii=False)+"\n")
        if not aligned or not bounded:
            raise RuntimeError("GRPO补采样本/奖励对应或次数验证失败；请查看resample-audit.jsonl")
        return result

    run._datarecipe_final_resample_fix = True
    GRPOTrainer._dynamic_sampling = run
    original.__globals__["logger"].info("DataRecipe: installed final-resample fix for ms-swift 4.0.1")


# Swift imports this file as an external plugin. Pure source helpers stay importable locally.
if __name__ != "recipe_sandbox.dandy.training.grpo_resample_patch":
    install()
