from groq import Groq
import os
import dotenv
from openai import OpenAI
import tiktoken 

dotenv.load_dotenv()



class Chatbot:
    def __init__(self):
        self.messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that provides information and answers questions in a short andconcise manner!.'
            }
        ]
        self.total_token = 0
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.groq_api_key)
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_limit = 5

        self.Grok = OpenAI(
            api_key=self.groq_api_key,
            base_url='https://api.groq.com/openai/v1'
        )


    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Chatbot, cls).__new__(cls)
        return cls.instance

    def bot(self, input_text):
        self.messages.append({"role": "user", "content": input_text})
        response = self.Grok.chat.completions.create(
        messages=self.messages, reasoning_effort="low",
        stream=True, stream_options={"include_usage": True},
        temperature=0, model="qwen/qwen3.8-27b",)

        response_chunks = []
        for chunk in response:
            if chunk.usage is not None:
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens
                self.total_token += chunk.usage.total_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content or ""
            response_chunks.append(content)
            print(content, end="")

        response_text = "".join(response_chunks)
        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

def main():
    chatbot = Chatbot()
    encoding = tiktoken.get_encoding("cl100k_base")
    # encoding = tiktoken.encoding_for_model("qwen/qwen3.8-27b")
    while True: 
        input_text = input("Ask something! ").strip().lower()

        token_count = len(encoding.encode(f"\n{input_text}"))

        if token_count > chatbot.token_limit:
            print(f"Input text exceeds the maximum token limit of {chatbot.token_limit} tokens. Your input has {token_count} tokens.")
            continue

        if input_text == 'exit':
            break

        chatbot.bot(input_text)
        



        print(f"\nTotal tokens used: {chatbot.total_token}")
        print(f"Input tokens used: {chatbot.input_tokens}")
        print(f"Output tokens used: {chatbot.output_tokens}")

if __name__ == "__main__":
    main()