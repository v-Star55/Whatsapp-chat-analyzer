import re 
import pandas as pd
from helper import get_sentiment_score

def preprocess(data):
    pattern ='\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:[ap]m)'
    
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

# Parse the date string into a datetime object
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p')

# Format the datetime object into the desired format
    df['message_date'] = df['message_date'].dt.strftime('%d/%m/%Y, %H:%M')

# changing the name
    df.rename(columns={'message_date':'date'},inplace=True)
    df = df.iloc[1:]

    df['user_message'] = df['user_message'].str.lstrip('- ')

# Split the 'user_message' column into 'user' and 'message'

    split_messages = df['user_message'].str.split(': ', expand=True)
    df['user'] = split_messages[0]
    df['message'] = split_messages[1]

# Replace empty 'user' values with 'Group Notification'
    df['user'] = df['user'].fillna('Group Notification')

# Drop the original 'user_message' column
    df.drop(columns=['user_message'], inplace=True)





    df['date'] = df['date'].str.replace(',', ' ')


# Extract the date and time from the 'date' column //////////////////////////////

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

# Extract year, month, day, hour, and minute
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_name'] = df['date'].dt.day_name()
    
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    
    # Apply sentiment analysis to each message
    df['sentiment_score'] = df['message'].apply(get_sentiment_score)
    
    return df