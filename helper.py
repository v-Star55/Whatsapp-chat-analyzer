from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

extract = URLExtract()

def get_sentiment_score(text):
    if isinstance(text, str):
        return sid.polarity_scores(text)['compound']
    else:
        return 0

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        if isinstance(message, str):
            words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        if isinstance(message, str):
            links.extend(extract.find_urls(message))

    # Perform sentiment analysis and calculate average sentiment score
    sentiment_scores = df['message'].apply(get_sentiment_score)
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    return num_messages, len(words), num_media_messages, len(links), avg_sentiment_score

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        if isinstance(message, str):
            for word in message.lower().split():
                if word not in stop_words:
                    y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        if isinstance(message, str):
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

from emoji import emoji_count

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        if isinstance(message, str):
            emojis.extend([c for c in message if emoji_count(c) > 0])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_trend_analysis(df):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group by date and calculate average sentiment score for each day
    sentiment_trend = df.groupby(df['date'].dt.date)['sentiment_score'].mean()

    return sentiment_trend



def most_busy_dates(df, top_n=5):
    busy_dates = df['only_date'].value_counts().nlargest(top_n)
    return busy_dates












