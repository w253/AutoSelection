"""Bailian transport for the existing search roles; no new agent workflow.

Keeps DataRecipe's streaming thinking mode and disables server-side web search.
Only the final answer is returned to the roles; reasoning is never logged.
Network errors propagate to the existing role/search retry logic.
"""
from recipe_sandbox.agents.base import LLMClient, ReasoningResponse, _split_thinking


class DandyLLMClient(LLMClient):
    """Apply run-level request options without changing the upstream client."""

    def __init__(self, config, key):
        """Configure transport once; SDK retries stay off as in DataRecipe."""
        super().__init__(api_key=key, base_url=config["base_url"], model=config["model"],
                         timeout=config.get("timeout_seconds", 1000), max_retries=0)
        self.options = dict(config)

    def chat_with_reasoning(self, prompt, temperature=0.6):
        """Stream thinking-enabled replies, returning only their final answer.

        Missing/null generation options are not sent. Configured temperature
        overrides role temperatures; omission retains the role's value.
        Optional thinking_budget/reasoning_effort require provider support.
        """
        options = self.options
        thinking = options.get("enable_thinking", False)
        extra = {"enable_search": False, "enable_thinking": thinking}
        if options.get("thinking_budget") is not None:
            extra["thinking_budget"] = options["thinking_budget"]
        arguments = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                         temperature=options.get("temperature") if options.get("temperature") is not None else temperature,
                         stream=thinking, extra_body=extra)
        for key in ("top_p", "max_tokens", "reasoning_effort"):
            if options.get(key) is not None:
                arguments[key] = options[key]
        response = self._client.chat.completions.create(**arguments)
        if thinking:
            pieces = []
            try:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        pieces.append(chunk.choices[0].delta.content)
            finally:
                response.close()
            content = "".join(pieces)
        else:
            content = response.choices[0].message.content or ""
        _, answer = _split_thinking(content)
        return ReasoningResponse(thinking="", answer=answer, raw=answer)

    def chat(self, prompt, temperature=0.6):
        """Expose the upstream plain-answer API through the same transport."""
        return self.chat_with_reasoning(prompt, temperature).answer
