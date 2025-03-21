from openai import OpenAI
from key import api_key
import time
import json

def request_story():
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    with open("pidgin.md", "r") as reference:
        context = reference.read()

    query = f"Using this following data as a reference, can you write a short children's story using this way of writing and exclusively this vocabulary: {context}?"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            temperature=1.5,  # Recommended for creative writing
            max_tokens=4000,   # Default max tokens
            stream=False
        )

        story_content = response.choices[0].message.content

        try:
            with open("response.json", "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(story_content)

        with open("response.json", "w") as file:
            json.dump(data, file, indent=4)

        print("Story generated and saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    for i in range(8):
        print(f"Requesting Story {i + 1}")
        request_story()
        time.sleep(5)

if __name__ == "__main__":
    main()