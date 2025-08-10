from openai import OpenAI

client = OpenAI(api_key="sk-0fqDBukOpggdWA1H5d3a3971270042A7968505D9Dc2f2848", base_url="https://api.shubiaobiao.cn/v1")

user_input = input("请输入你的问题：")
completion = client.chat.completions.create(
    model="claude-sonnet-4-20250514-thinking",
    stream=False,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(completion.choices[0].message)