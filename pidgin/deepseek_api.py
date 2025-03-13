from openai import OpenAI
from key import api_key
import json

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

reference = open("pidgin.md", "r")
context = reference.read()

query = f"Using this following data as a reference, can you write a short childrens story using this way of writing: {context}?"

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": query},
    ],
    # Deepseek documentation recommends 1.5 for creative writing
    temperature = 1.5,
    # Max tokens defaults to 4k
    max_tokens = 4000,
    stream=False
)

with open("response.json", "w") as file:
    json.dump([str(response.choices[0].message.content)], file)
print(response)