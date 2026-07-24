import time
import os
from datetime import datetime

import anthropic

from ..eval_types import MessageList, SamplerBase, SamplerResponse
from .. import common

CLAUDE_SYSTEM_MESSAGE_LMSYS = (
    "The assistant is Claude, created by Anthropic. The current date is "
    f"{datetime.now().strftime('%B %d, %Y')}. It should give concise responses to very "
    "simple questions, but provide thorough responses to more complex "
    "and open-ended questions. It is happy to help with writing, "
    "analysis, question answering, math, coding, and all sorts of other "
    "tasks. It uses markdown for coding. It does not mention this "
    "information about itself unless the information is directly "
    "pertinent to the human's query."
)
# reference: https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894


class ClaudeCompletionSampler(SamplerBase):

    def __init__(
        self,
        model: str,
        system_message: str | list | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        thinking: dict | None = None,
    ):
        self.client = anthropic.Anthropic()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.image_format = "base64"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }
        return new_image

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                if not common.has_only_user_assistant_messages(message_list):
                    raise ValueError(f"Claude sampler only supports user and assistant messages, got {message_list}")

                kwargs: dict = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": message_list,
                }

                if self.system_message is not None:
                    kwargs["system"] = self.system_message

                if self.thinking is not None:
                    # Extended thinking requires temperature=1.0; ignore any explicit temperature setting
                    kwargs["thinking"] = self.thinking
                elif self.temperature is not None:
                    kwargs["temperature"] = self.temperature

                response_message = self.client.messages.create(**kwargs)

                # Handle responses that may include thinking blocks (extended thinking)
                response_text = next(
                    (block.text for block in response_message.content if hasattr(block, "text")),
                    "",
                )

                # Safety classifiers (e.g. on claude-fable-5) can decline with
                # stop_reason "refusal"; record it so empty responses are auditable.
                error = None
                if response_message.stop_reason == "refusal":
                    details = getattr(response_message, "stop_details", None)
                    category = getattr(details, "category", None) if details else None
                    error = f"refusal: {category or 'unspecified'}"

                claude_input_messages: MessageList = message_list
                if self.system_message:
                    claude_input_messages = [{"role": "system", "content": self.system_message}] + message_list

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"error": error},
                    actual_queried_message_list=claude_input_messages,
                )
            except anthropic.RateLimitError as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
