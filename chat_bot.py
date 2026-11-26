from groq import Groq
import os
import dotenv
from openai import OpenAI
import tiktoken 
from transformers import AutoTokenizer


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
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = Groq(api_key=self.groq_api_key)
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_limit = 50
        self.iteration = 0

        self.Grok = OpenAI(
            api_key=self.groq_api_key,
            base_url='https://api.groq.com/openai/v1'
        )


    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Chatbot, cls).__new__(cls)
        return cls.instance

    def trim_messages(self, max_number=4):
        if len(self.messages) < max_number:
            return

        system_messages = [msg for msg in self.messages if msg['role'] == 'system']
        other_messages = [msg for msg in self.messages if msg['role'] != 'system']

        trimmed_messages = other_messages[-max_number:]  # Keep the last max_number messages
        self.messages = [*system_messages,  *trimmed_messages]

        # print(f"\n Trimmed messages to the last {max_number} messages. Messages {self.messages} \nCurrent message count: {len(self.messages)}")

        return self.messages


    def summarize_messages(self):
        if self.iteration % 5 != 0:
            # Only summarize every 5 iterations to reduce token usage
            print(f"Skipping summarization for iteration {self.iteration}.")
            return

        conversation_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in self.messages
        )

        response = self.Grok.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the conversation concisely, preserving important facts and decisions.",
                },
                {
                    "role": "user",
                    "content": f"Please summarize this conversation:\n\n{conversation_text}",
                },
            ],
            reasoning_effort="low",
            stream=False,
            temperature=0,
            model="qwen/qwen3.8-27b",
        )

        summary_text = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": summary_text})
        print(f"\nSummary of the conversation:\n{summary_text}\n")

        # Keep only the last 4 messages after summarization
        self.trim_messages(max_number=4)
        print('Summarization was successful')

        return summary_text
        

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

        self.summarize_messages()

        return response_text

def main():
    chatbot = Chatbot()
    # encoding = tiktoken.get_encoding("cl100k_base")
    encoding = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', use_auth_token=chatbot.hf_token)

    while True: 
        chatbot.iteration += 1
        input_text = input("Ask something! ").strip().lower()

        token_count = len(encoding.encode(f"\n{input_text}"))

        if token_count > chatbot.token_limit:
            print(f"Input text exceeds the maximum token limit of {chatbot.token_limit} tokens. Your input has {token_count} tokens.")
            continue

        if not input_text:
            print("Input text cannot be empty. Please enter a valid question.")
            continue

        # if chatbot.total_token >= 20:
        #     print("Total token limit reached. Please restart the program to continue.")
        #     break

        if input_text == 'exit':
            break

        chatbot.bot(input_text)
        



        print(f"\nTotal tokens used: {chatbot.total_token}")
        print(f"Input tokens used: {chatbot.input_tokens}")
        print(f"Output tokens used: {chatbot.output_tokens}")

if __name__ == "__main__":
    main()