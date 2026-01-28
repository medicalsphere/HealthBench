import os
import time
from typing import Any

from google import genai

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class GeminiSampler(SamplerBase):
    """
    Sample from Google's Gemini API
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "GEMINI_API_KEY"
        assert os.environ.get("GEMINI_API_KEY"), "Please set GEMINI_API_KEY"
        self.client = genai.Client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _sanitize_utf8(self, text: str) -> str:
        """
        Sanitize text to ensure valid UTF-8 encoding.
        Removes null bytes, replacement characters, and invalid UTF-8 sequences.
        """
        if not text:
            return text
        
        try:
            # Remove null bytes and replacement characters
            text = text.replace('\0', '').replace('\ufffd', '')
            
            # Encode to UTF-8 and decode back to catch any invalid sequences
            # Use 'ignore' to skip invalid bytes
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            
            return text
        except Exception as e:
            print(f"Warning: UTF-8 sanitization failed: {e}")
            # Fallback: remove non-ASCII characters if encoding fails
            return ''.join(char for char in text if ord(char) < 128)

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _convert_messages_to_gemini_format(self, message_list: MessageList) -> list[dict[str, Any]]:
        """
        Convert MessageList format to Gemini's contents format.
        Gemini uses 'user' and 'model' roles with parts structure.
        """
        gemini_messages = []
        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            # Skip system messages as they're handled separately
            if role == "system" or role == "developer":
                continue

            gemini_role = "model" if role == "assistant" else "user"

            # Handle content as string or list
            if isinstance(content, str):
                gemini_messages.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
            elif isinstance(content, list):
                # Content is a list of parts - concatenate text parts
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "input_text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    gemini_messages.append({
                        "role": gemini_role,
                        "parts": [{"text": " ".join(text_parts)}]
                    })

        return gemini_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Convert messages to Gemini format
        conversation_history = self._convert_messages_to_gemini_format(message_list)

        # Track actual queried messages
        actual_message_list = message_list.copy()
        if self.system_message:
            actual_message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        trial = 0
        while True:
            try:
                # Use generate_content API for multi-turn conversation support
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=conversation_history,
                )

                # Extract response text and sanitize it
                response_text = response.text
                response_text = self._sanitize_utf8(response_text)

                # Extract usage metadata
                usage_metadata = getattr(response, 'usage_metadata', None)

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"usage": usage_metadata},
                    actual_queried_message_list=actual_message_list,
                )
            except Exception as e:
                # Handle rate limits and other errors with exponential backoff
                exception_backoff = 2**trial
                print(
                    f"Exception occurred, retrying {trial} after {exception_backoff} sec: {e}"
                )
                time.sleep(exception_backoff)
                trial += 1

                # Limit retries to avoid infinite loop
                if trial > 5:
                    print(f"Max retries exceeded. Returning empty response.")
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None},
                        actual_queried_message_list=actual_message_list,
                    )