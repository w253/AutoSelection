"""Unified schema-mapping agent implementation."""

from __future__ import annotations

import hashlib
import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from tqdm import tqdm

from recipe_sandbox.agents.base import LLMClient
from recipe_sandbox.schema.enums import Split

_SCHEMA_CONFIG_PATH = Path(__file__).resolve().parent.parent / "schema" / "canonical_schema.yaml"
_ALLOWED_IMPORTS = {"json", "hashlib", "copy", "re"}
_REQUIRED_TOP_KEYS = {"sample_id", "source_name", "messages"}


def load_schema_config(path: Optional[str] = None) -> str:
    config_path = Path(path) if path else _SCHEMA_CONFIG_PATH
    return config_path.read_text(encoding="utf-8")


def build_mapping_prompt(
    sample_records: List[dict],
    schema_config_text: str,
    source_name: str = "unknown",
) -> str:
    samples_json = json.dumps(sample_records[:5], indent=2, ensure_ascii=False)
    return textwrap.dedent(f"""\
You are a data-engineering assistant. Your task is to write a **single Python function** called `map_record` that converts a raw JSON record (given as a Python dict) into a dict that conforms to the CanonicalSample schema described below.

## Target Schema
```yaml
{schema_config_text}
```

## Sample Input Records (up to 5)
```json
{samples_json}
```

## Requirements
1. The function signature MUST be:
   ```python
   def map_record(record: dict, source_name: str = "{source_name}") -> dict:
   ```
2. Return a dict with the keys expected by CanonicalSample.
3. For `sample_id`: if the source has a unique id field use it (as a string). Otherwise compute `hashlib.md5(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()`.
4. For `messages`: figure out the conversation structure. Common patterns:
   - `"messages"` or `"conversations"` key containing a list of {{role, content}} dicts.
   - `"instruction"` + `"input"` + `"output"` (Alpaca format) -> user message = instruction + input, assistant message = output.
   - `"prompt"` + `"response"` / `"completion"` -> user message + assistant message.
   - `"question"` + `"answer"` -> user message + assistant message.
   - IMPORTANT MULTIPLE-CHOICE RULE: if the record has fields like `"question"`, `"choices"` (a list), and a numeric `"answer"`, then ALL choices must be appended to the user prompt, and `target.text` must preserve the raw answer index/label (e.g. `"2"`), NOT the answer option content.
   - If a `"system"` or `"system_prompt"` field exists, prepend a system message.
   IMPORTANT: If you inject explanation/reasoning into the final answer, the final `assistant` message content MUST exactly match the `target.text` (including the explanations).
5. For `target`: if there is an output/answer/response field, put it in `target.text`. IMPORTANT: If there are reasoning/explanation fields (like "explanation", "reasoning", "cot"), inject them PRECEDING the final answer in `target.text` (e.g. `explanation + "\n\nAnswer: " + answer`) so the model learns Chain-of-Thought. DO NOT put explanation/reasoning in `metadata` or as a system prompt.
6. For `metadata`: infer `task_type` if obvious, otherwise use `"other"`. Set `has_system_prompt` and `has_multiturn` accordingly. Put any other extra fields from the source record that do not map to standard fields into `metadata.extra` (but NOT explanations).
7. For fields not present in the source, use sensible defaults (empty string, empty list, None, etc.).
8. Only use Python standard library (`json`, `hashlib`, `copy`). No external imports.
9. The function must be self-contained.
10. Avoid backslashes inside f-string expressions. If a source field name contains an apostrophe, do not write expressions like `record['Writer\\'s X']` inside `{...}`.
11. If you set `metadata.source_split`, it must be one of exactly: `"train"`, `"dev"`, `"test"`, or `"unspecified"`.

## Output Format
Return ONLY the Python function inside a ```python ... ``` code block. No explanation outside the block.
""")


def extract_code_block(llm_response: str) -> str:
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = llm_response.strip()
    if stripped.startswith("def "):
        return stripped
    raise ValueError("Could not extract a Python code block from the LLM response.")


def _validate_code_safety(code: str) -> None:
    for line in code.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("import ") or line_stripped.startswith("from "):
            if line_stripped.startswith("from "):
                mod = line_stripped.split()[1].split(".")[0]
            else:
                mod = line_stripped.split()[1].split(".")[0]
            if mod not in _ALLOWED_IMPORTS:
                raise ValueError(f"Unsafe import detected: '{mod}'. Only {_ALLOWED_IMPORTS} are allowed.")


def sanitize_mapping_code(code: str) -> str:
    """Repair common LLM-generated Python issues before compilation.

    The most common failure mode is using escaped apostrophes inside f-string
    expressions, e.g.:
        f"{record['Writer\\'s Difficulty Estimate']}"
    Python rejects backslashes inside the expression section of f-strings.

    We normalize any single-quoted string literal containing an escaped
    apostrophe into a backslash-free concatenation expression, preserving the
    semantic value even inside f-string expressions.
    """

    pattern = re.compile(r"'((?:[^'\\]|\\.)*\\'(?:[^'\\]|\\.)*)'")

    def repl(match: re.Match[str]) -> str:
        literal = match.group(0)
        try:
            value = ast.literal_eval(literal)
        except Exception:
            return literal
        if not isinstance(value, str):
            return literal
        parts = value.split("'")
        quoted_parts = [repr(part) for part in parts]
        separator = " + chr(39) + "
        return "(" + separator.join(quoted_parts) + ")"

    return pattern.sub(repl, code)


def compile_mapping_fn(code: str) -> Callable[[dict, str], dict]:
    code = sanitize_mapping_code(code)
    _validate_code_safety(code)
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102
    except SyntaxError as exc:
        raise ValueError(f"Generated mapping code has syntax error: {exc}") from exc
    fn = namespace.get("map_record")
    if fn is None:
        raise ValueError("The generated code does not define a 'map_record' function.")
    if not callable(fn):
        raise TypeError("'map_record' is not callable.")
    return fn


def validate_mapping_output(output: dict) -> List[str]:
    problems: List[str] = []
    for key in _REQUIRED_TOP_KEYS:
        if key not in output:
            problems.append(f"Missing required key: '{key}'")
    if "messages" in output:
        msgs = output["messages"]
        if not isinstance(msgs, list) or len(msgs) == 0:
            problems.append("'messages' must be a non-empty list")
        else:
            valid_roles = {"system", "user", "assistant", "tool"}
            for i, message in enumerate(msgs):
                if not isinstance(message, dict):
                    problems.append(f"messages[{i}] is not a dict")
                    continue
                if "role" not in message:
                    problems.append(f"messages[{i}] missing 'role'")
                elif message["role"] not in valid_roles:
                    problems.append(f"messages[{i}] invalid role: '{message['role']}'")
                if "content" not in message:
                    problems.append(f"messages[{i}] missing 'content'")
    metadata = output.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            problems.append("'metadata' must be a dict when present")
        else:
            split_value = metadata.get("source_split")
            if split_value is not None:
                try:
                    Split(split_value)
                except ValueError:
                    problems.append(
                        "metadata.source_split must be one of "
                        f"{', '.join(split.value for split in Split)}; got '{split_value}'"
                    )
    return problems


def validate_fn_on_samples(
    fn: Callable,
    sample_records: List[dict],
    source_name: str = "unknown",
) -> List[str]:
    all_problems: List[str] = []
    for i, record in enumerate(sample_records):
        try:
            result = fn(record, source_name)
        except Exception as exc:
            all_problems.append(f"Record {i}: exception -> {exc}")
            continue
        if not isinstance(result, dict):
            all_problems.append(f"Record {i}: returned {type(result)}, expected dict")
            continue
        problems = validate_mapping_output(result)
        for problem in problems:
            all_problems.append(f"Record {i}: {problem}")
        all_problems.extend(_validate_special_cases(record, result, i))
    return all_problems


def _is_numeric_multiple_choice_record(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    question = record.get("question")
    choices = record.get("choices")
    answer = record.get("answer")
    if not isinstance(question, str) or not question.strip():
        return False
    if not isinstance(choices, list) or not choices or not all(isinstance(item, str) for item in choices):
        return False
    if isinstance(answer, bool) or answer is None:
        return False
    return isinstance(answer, (int, float))


def _normalize_mcq_answer(answer: Any) -> str:
    if isinstance(answer, float) and answer.is_integer():
        return str(int(answer))
    if isinstance(answer, int):
        return str(answer)
    if answer is None:
        return ""
    return str(answer).strip()


def _validate_special_cases(record: dict, result: dict, record_index: int) -> List[str]:
    problems: List[str] = []
    if not _is_numeric_multiple_choice_record(record):
        return problems

    messages = result.get("messages") or []
    user_content = ""
    if messages and isinstance(messages[0], dict):
        user_content = str(messages[0].get("content", ""))

    for choice in record.get("choices", []):
        if choice not in user_content:
            problems.append(
                f"Record {record_index}: multiple-choice mapping must include every choice in the user prompt; missing choice '{choice[:80]}'"
            )
            break

    target = result.get("target") or {}
    target_text = str(target.get("text", "")).strip()
    expected = _normalize_mcq_answer(record.get("answer"))
    if target_text != expected:
        problems.append(
            f"Record {record_index}: multiple-choice mapping must keep target.text as the raw answer index/label '{expected}', got '{target_text}'"
        )

    return problems


class AgentMapper:
    """Use an LLM to auto-generate a mapping function for arbitrary JSONL data."""

    def __init__(
        self,
        llm_client: Any = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        schema_config_path: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        if llm_client is not None:
            self._llm = llm_client
        else:
            self._llm = LLMClient(api_key=api_key, base_url=base_url, model=model)
        self._schema_text = load_schema_config(schema_config_path)
        self._max_retries = max_retries
        self._mapping_fn: Optional[Callable] = None
        self._mapping_code: Optional[str] = None

    def generate_mapping(
        self,
        sample_records: List[dict],
        source_name: str = "unknown",
    ) -> Callable[[dict, str], dict]:
        prompt = build_mapping_prompt(sample_records, self._schema_text, source_name)
        last_error: Optional[str] = None
        print("[AgentMapper] Generating mapping function with LLM...")
        for attempt in range(1 + self._max_retries):
            if attempt > 0 and last_error:
                prompt_with_fix = (
                    prompt
                    + f"\n\n## Previous attempt failed with these errors:\n{last_error}\n"
                    "Please fix the issues and return the corrected function."
                )
            else:
                prompt_with_fix = prompt

            raw_response = self._call_llm(prompt_with_fix)
            try:
                code = extract_code_block(raw_response)
                code = sanitize_mapping_code(code)
                fn = compile_mapping_fn(code)
            except (ValueError, TypeError) as exc:
                last_error = str(exc)
                continue
            print("[AgentMapper] Validating the generated function on sample records...")
            problems = validate_fn_on_samples(fn, sample_records, source_name)
            if not problems:
                self._mapping_fn = fn
                self._mapping_code = code
                return fn
            last_error = "\n".join(problems)

        raise RuntimeError(
            f"Failed to generate a valid mapping after {1 + self._max_retries} attempts. "
            f"Last errors:\n{last_error}"
        )

    def map_record(self, record: dict, source_name: str = "unknown") -> dict:
        if self._mapping_fn is None:
            raise RuntimeError("Call generate_mapping() first.")
        return self._mapping_fn(record, source_name)

    @property
    def mapping_code(self) -> Optional[str]:
        return self._mapping_code

    def save_mapping_code(self, path: str) -> None:
        if self._mapping_code is None:
            raise RuntimeError("No mapping code generated yet.")
        Path(path).write_text(self._mapping_code, encoding="utf-8")

    def load_mapping_code(self, path: str) -> Callable[[dict, str], dict]:
        code = Path(path).read_text(encoding="utf-8")
        fn = compile_mapping_fn(code)
        self._mapping_fn = fn
        self._mapping_code = code
        return fn

    def read_jsonl(
        self,
        path: str,
        source_name: str = "unknown",
        n_sample: int = 5,
    ) -> Iterator[dict]:
        records = _read_raw_jsonl(path)
        print(f"[AgentMapper] Loaded {len(records)} records from {path}")
        if not records:
            return
        print(f"[AgentMapper] Sample record for inspection:\n{json.dumps(records[0], indent=2, ensure_ascii=False)}")
        sampled = records[:n_sample]
        if not sampled:
            return
        if self._mapping_fn is None:
            self.generate_mapping(sampled, source_name)
        else:
            cached_problems = validate_fn_on_samples(self._mapping_fn, sampled, source_name)
            if cached_problems:
                print("[AgentMapper] Cached mapping failed validation. Regenerating with LLM...")
                self._mapping_fn = None
                self._mapping_code = None
                self.generate_mapping(sampled, source_name)

        for record in tqdm(records, desc="Mapping records", unit="record"):
            yield self.map_record(record, source_name)

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self._llm, "chat"):
            return self._llm.chat(prompt)
        return self._llm(prompt)


def _read_raw_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stable_hash(data: dict) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(text.encode()).hexdigest()
