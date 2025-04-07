# EdenAI
Edenai is a full-stack AI platform for developers to efficiently create, test, and deploy AI: unified access to the best AI models combined with a powerful workflow builder and monitoring tools.
for more informations about the models available visit : https://www.edenai.co

To use EdenAI with `aisuite`, you’ll need an [EdenAI account](https://www.edenai.co). After logging in, go to the [API Keys](https://app.edenai.run/admin/api-settings/features-preferences) section in your account settings and copy your api key. Once you have your key, add it to your environment as follows:

```shell
export EDENAI_API_KEY="your-edenai-api-key"
```

## Create a Chat Completion

Install the `edenai` Python client:

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "edenai"
# edenai provide a wide varaity of models, for more informations about the models available visit : https://docs.edenai.co/reference/llm_llm_chat_create
model_id = "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

EdenAI alse supports multimodal calls : 

```python
import aisuite as ai
client = ai.Client()

provider = "edenai"
model_id = "openai/gpt-4o"

# for more informations about the format of the messages visit : https://docs.edenai.co/reference/llm_llm_chat_create
messages = [
        {"role": "user", "content":[
            {
                "type": "text",
                "text": "what is the emotion in this man's face?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQA2xNb4xY7_pK886sVo7JhjAdXxvch6zXIMg&s"
                              }
            }
]}

]
```
Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
