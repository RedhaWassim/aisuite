from unittest.mock import MagicMock, patch
import pytest
from aisuite.providers.edenai_provider import EdenaiProvider

@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")

def test_edenai_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = EdenaiProvider()
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        selected_model: {
            "generated_text": response_text_content
        }
    }

    with patch('httpx.post', return_value=mock_response) as mock_post:
        response = provider.chat_completions_create(
            model=selected_model,
            messages=message_history,
            temperature=chosen_temperature,
        )

        mock_post.assert_called_with(
            provider.BASE_URL,
            json={
                "providers": [selected_model],
                "messages": provider._reformat_multimodal(message_history),
                "temperature": chosen_temperature
            },
            headers={
                "Authorization": f"Bearer test-api-key",
                "Content-Type": "application/json"
            },
            timeout=provider.timeout
        )

        assert response.choices[0].message.content == response_text_content
