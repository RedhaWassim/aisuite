import os
import httpx
import json
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from typing import List


class EdenaiProvider(Provider):
    BASE_URL = "https://api.edenai.run/v2/llm/chat"

    def __init__(self, **config):
        self.api_key = config.get("api_key", os.getenv("EDENAI_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "EDENAI_API_KEY is required in config or environment variables"
            )
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model: str, messages: List[dict], **kwargs):
        model_name = model.split(":")[-1]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


        payload = {"model": model_name, "messages": messages, **kwargs}

        try:
            response = httpx.post(
                self.BASE_URL, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"EdenAI API error: {e.response.text}") from e

        return self._normalize_response(response.json())

    def _normalize_response(self, response_data: dict) -> ChatCompletionResponse:
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return normalized_response
