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

    def _reformat_multimodal(self, messages: List[dict]) -> List[dict]:
        reformatted = []
        for message in messages:
            new_msg = {"role": message["role"], "content": []}
            content = message.get("content", "")

            if isinstance(content, list):
                new_msg["content"] = self._transform_content_parts(content)
            else:
                try:
                    parsed = json.loads(content)
                    new_msg["content"] = (
                        self._transform_content_parts(parsed)
                        if isinstance(parsed, list)
                        else [{"type": "text", "text": str(parsed)}]
                    )
                except (json.JSONDecodeError, TypeError):
                    new_msg["content"].append({"type": "text", "text": str(content)})

            reformatted.append(new_msg)
        return reformatted

    def _transform_content_parts(self, parts: List[dict]) -> List[dict]:
        transformed = []
        for part in parts:
            part_type = part.get("type")
            content = part.get("content", {})

            if part_type == "text":
                transformed.append({"type": "text", "text": content.get("text", "")})
            elif part_type == "media_url":
                transformed.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": content.get("media_url", "")},
                    }
                )
            else:
                transformed.append(part)
        return transformed

    def chat_completions_create(self, model: str, messages: List[dict], **kwargs):
        model_name = model.split(":")[-1]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        formatted_messages = self._reformat_multimodal(messages)

        payload = {"model": model_name, "messages": formatted_messages, **kwargs}

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
