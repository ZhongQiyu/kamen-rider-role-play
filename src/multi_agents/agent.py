# agent.py

from textblob import TextBlob

class NlpAgent:
    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def feedback_loop(self, user_feedback, text):
        print(f"Received user feedback: {user_feedback} for text: {text}")

if __name__ == "__main__":
    # NLP 情感分析
    agent = NlpAgent()
    text = "I love this movie. It's amazing!"
    sentiment = agent.analyze_sentiment(text)
    print(f"Sentiment polarity: {sentiment}")
    user_feedback = "positive" if sentiment > 0 else "negative"
    agent.feedback_loop(user_feedback, text)
