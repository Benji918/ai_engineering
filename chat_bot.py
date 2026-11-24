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

def bot(input_text):
    response = Grok.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"input: {input_text}\n\nPlease provide a detailed and informative response.",
        }
    ],
    model="qwen/qwen3.8-27b",
)
    return response.choices[0].message.content


def main():
    input_text = input("Ask something! ")
    print(bot(input_text))

if __name__ == "__main__":
    main()