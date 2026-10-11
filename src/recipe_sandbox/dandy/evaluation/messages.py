"""图文 benchmark 提示词；保留 formatter 文本，不追加答案或答案指令。"""


def vlm_text_input(record, params):
    """返回 SWIFT InferRequest 可用的消息与有序图片。"""
    prompt = "\n".join("<image>" for _ in record["images"]) + "\n\n" + record["text_input"]
    return {"messages": [{"role": "user", "content": prompt}], "images": list(record["images"])}
