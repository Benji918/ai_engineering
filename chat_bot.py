from groq import Groq
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=groq_api_key)

Grok = OpenAI(
    api_key=groq_api_key,
    base_url='https://api.groq.com/openai/v1'
)

messages = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant that provides information and answers questions in a short andconcise manner!.'
    }
]

def bot(input_text):
    messages.append({"role": "user", "content": input_text})
    response = Grok.chat.completions.create(
    messages=messages, reasoning_effort="low",
    stream=True,temperature=0, model="qwen/qwen3.8-27b",)

    response_chunks = []
    for chunk in response:
        response_chunks.append(chunk.choices[0].delta.content or "")
        print(chunk.choices[0].delta.content or "", end="")

    response_text = "".join(response_chunks)
    messages.append({"role": "assistant", "content": response_text})

def main():
    while True: 
        input_text = input("Ask something! ").strip().lower()

        if input_text == 'exit':
            break

        bot(input_text)
        print()

if __name__ == "__main__":
    main()