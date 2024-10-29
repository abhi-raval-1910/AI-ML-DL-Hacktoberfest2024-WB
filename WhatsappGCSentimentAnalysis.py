import pandas as pd
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load WhatsApp chat data
def load_chat_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        chat_data = file.readlines()
    return chat_data

# Parse chat data into a DataFrame
def parse_chat_data(chat_data):
    messages = []
    for line in chat_data:
        # Match pattern for WhatsApp export format (e.g., "12/25/20, 9:08 PM - Name: Message")
        match = re.match(r"(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2} (?:AM|PM)) - (.*?): (.*)", line)
        if match:
            date, time, sender, message = match.groups()
            messages.append([date, time, sender, message])
    # Create DataFrame
    return pd.DataFrame(messages, columns=["Date", "Time", "Sender", "Message"])

# Perform sentiment analysis using TextBlob
def analyze_sentiment_textblob(message):
    blob = TextBlob(message)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Perform sentiment analysis using Vader
def analyze_sentiment_vader(message, analyzer):
    scores = analyzer.polarity_scores(message)
    polarity = scores['compound']
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Analyze chat sentiment
def analyze_chat_sentiment(df, method="vader"):
    analyzer = SentimentIntensityAnalyzer() if method == "vader" else None
    sentiments = []
    
    for message in df['Message']:
        if method == "textblob":
            sentiment, polarity = analyze_sentiment_textblob(message)
        elif method == "vader":
            sentiment, polarity = analyze_sentiment_vader(message, analyzer)
        sentiments.append((sentiment, polarity))

    # Add sentiment data to DataFrame
    df['Sentiment'], df['Polarity'] = zip(*sentiments)
    return df

# Run sentiment analysis
def main():
    filepath = 'path_to_your_chat_file.txt'  # Replace with your file path
    chat_data = load_chat_data(filepath)
    chat_df = parse_chat_data(chat_data)
    
    print("Starting sentiment analysis...")
    chat_df = analyze_chat_sentiment(chat_df, method="vader")  # Use "textblob" or "vader" as method
    
    # Show the results
    print(chat_df.head())

    # Summary of sentiment
    sentiment_summary = chat_df['Sentiment'].value_counts()
    print("\nSentiment Summary:")
    print(sentiment_summary)

    # Average polarity for each sender
    polarity_by_sender = chat_df.groupby("Sender")["Polarity"].mean()
    print("\nAverage Polarity by Sender:")
    print(polarity_by_sender)

if __name__ == "__main__":
    main()
