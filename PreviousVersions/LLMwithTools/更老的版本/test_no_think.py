import os
# 移除代理环境变量
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen3.5-9B",
    messages=[
        {"role": "system", "content": "你是个猫娘"},
        { "role": "user", "content": "你好" }
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096,
)

print(response)
