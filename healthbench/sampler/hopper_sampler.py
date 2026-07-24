import asyncio
import os
import time

import hopper

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class HopperSampler(SamplerBase):
    """
    Sampler for models accessible via the hopper unified API client.
    Each instance is bound to one model and one API key env var.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        max_tokens: int | None = 4096,
        provider: str | None = None,
        reasoning: dict | None = None,
        provider_options: dict | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider
        self.reasoning = reasoning
        self.provider_options = provider_options
        api_key = os.environ.get(api_key_env)
        assert api_key, f"Please set {api_key_env}"
        self.credentials = hopper.Credentials(api_key=api_key)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        messages = [
            hopper.CanonicalMessage(role=m["role"], content=m["content"])
            for m in message_list
        ]
        request = hopper.CanonicalRequest(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            provider=self.provider,
            reasoning=self.reasoning,
            provider_options=self.provider_options or {},
        )

        trial = 0
        while True:
            try:
                envelope = asyncio.run(hopper.complete(request, self.credentials))
                # Safety classifiers (e.g. claude-fable-5) can decline with a
                # "refusal" finish reason; record it so empty responses are auditable.
                error = None
                if envelope.response.finish_reason == "refusal":
                    error = "refusal"
                return SamplerResponse(
                    response_text=envelope.response.content,
                    response_metadata={"usage": envelope.usage, "error": error},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                backoff = 2**trial
                print(f"Exception, retrying {trial} after {backoff}s: {e}")
                time.sleep(backoff)
                trial += 1
                if trial > 5:
                    return SamplerResponse(
                        response_text="",
                        response_metadata={
                            "usage": None,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        actual_queried_message_list=message_list,
                    )
