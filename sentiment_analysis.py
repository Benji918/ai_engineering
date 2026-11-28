import os
import dotenv
from openai import OpenAI
from groq import Groq
from transformers import pipeline

dotenv.load_dotenv()

class Analyser:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = Groq(api_key=self.groq_api_key)

        self.Grok = OpenAI(
            api_key=self.groq_api_key,
            base_url='https://api.groq.com/openai/v1'
        )

        self.sentiment_analysis_pipeline = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')

    def analyze_sentiment(self, text):
        sentiment_result = self.sentiment_analysis_pipeline(
            text,
            candidate_labels=["positive", "negative", "neutral"],
        )
        return sentiment_result


def main():
    analyser = Analyser()
    text = "I really like this product!"
    sentiment = analyser.analyze_sentiment(text)
    print(sentiment)




if '__main__' == __name__:
    main()