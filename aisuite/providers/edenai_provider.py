import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from typing import List
import json

class EdenaiProvider(Provider):
    """
    EdenAI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.edenai.run/v2/multimodal/chat"

    def __init__(self, **config):
        """
        Initialize the EdenAI provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("EDENAI_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "EDENAI API key is missing. Please provide it in the config or set the EDENAI_API_KEY environment variable."
            )

        self.timeout = config.get("timeout", 30)

    def _is_multimodal_request(self, messages: List[dict]) -> bool:
        """
        Check if the request is a multimodal request by inspecting the content of the messages.

        Args:
            messages (List[dict]): The list of messages to check.

        Returns:
            bool: True if the request is multimodal, False otherwise.
        """
        for msg in messages:
            content_list = msg.get("content", [])
            if isinstance(content_list, list):
                for item in content_list:
                    if (
                        isinstance(item, dict)
                        and "type" in item
                        and item["type"] in ["text", "media_url", "media_base64"]
                    ):
                        return True
        return False
    def _reformat_multimodal(self, messages):
        """
        Reformat messages into multimodal format if not already,
        or extract the content properly if already in multimodal form.

        Args:
            messages (List[dict]): The list of messages.

        Returns:
            List[dict]: The reformatted or extracted messages.
        """
        reformatted = []

        for message in messages:
            new_message = {"role": message["role"], "content": []}
            content = message.get("content", "")

            try:
                parsed_content = json.loads(content)
                if self._is_multimodal_request([{"content": parsed_content}]):
                    new_message["content"].extend(parsed_content)
                else:
                    new_message["content"].append({
                        "type": "text",
                        "content": {"text": content}
                    })
            except json.JSONDecodeError:
                new_message["content"].append({
                    "type": "text",
                    "content": {"text": content}
                })

            reformatted.append(new_message)

        return reformatted

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the EDENAI multimodal chat endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = self._reformat_multimodal(messages)
        data = {
                "providers": [model],
                "messages": messages,
                **kwargs,   
                }
        try:
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"EdenAI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        return self._normalize_response(response.json(), model)

    def _normalize_response(self, response_data, model):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        provider_response = response_data.get(model, {})
        response_content = provider_response.get("generated_text", "")
        
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].model = model
        normalized_response.choices[0].message.content = response_content
        return normalized_response
