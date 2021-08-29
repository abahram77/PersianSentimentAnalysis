from  Dataset import x,y


import getpass

APIKEY = getpass.getpass()


# running Translate API
def translate_persian2english(list_of_sentences_in_english):
    from googleapiclient.discovery import build
    service = build('translate', 'v2', developerKey=APIKEY)

    # use the service
    inputs = list_of_sentences_in_english
    outputs = service.translations().list(source='fa', target='en', q=inputs).execute()
    # print outputs
    return [i['translatedText'] for i in outputs['translations']]


translate_persian2english(["فک کنم پاسارگاد یا بانک سامان را دوست داشته باشم"])

import os

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = "/content/drive/MyDrive/idehhub/SentimentGCkey/bamboo-sweep-320713-65916259c80a.json"

from google.cloud import language_v1
from google.cloud.language_v1 import enums


def sample_analyze_sentiment(text_content):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_sentiment(document, encoding_type=encoding_type)
    # Get overall sentiment of the input document
    return (response.document_sentiment.score)


def delete_punctuation_marks(text):
    new_string = ""
    persian_punctuations = ['…', '،']
    for i in text:
        if (i not in string.punctuation and i not in persian_punctuations):
            new_string += i
    return new_string


def delete_at_signs_for_list_of_just_tweets(data):
    new_data = []
    import string
    for i in range(len(data)):
        temp = data[i].split()
        new_tweet_list = []
        for j in range(len(temp)):
            if (temp[j][0] not in string.punctuation):
                new_tweet_list.append(temp[j])
        new_tweet = ' '.join(new_tweet_list)
        new_data.append(new_tweet)
    return new_data


def delete_at_signs(data):
    new_data = []
    for i in range(len(data)):
        temp = data[i][0].split()
        new_tweet_list = []
        for j in range(len(temp)):
            if (temp[j][0] != '@'):
                new_tweet_list.append(temp[j])
        new_tweet = ' '.join(new_tweet_list)
        new_data.append([new_tweet, data[i][1]])
    return new_data


def filter_tweets_by_keywords(keywords, path, from_date, to_date):
    tweets = get_between_dates_list_of_tweets(from_date, to_date, path)
    tweets = [tweet[0] for tweet in tweets]
    tweets = delete_at_signs_for_list_of_just_tweets(tweets)
    tweets = delete_other_than_persian_for_list_of_just_tweets(tweets)
    filterd_tweets = []
    for tweet in tweets:
        list_of_words_in_tweet = tweet.split(' ')
        for keyword in keywords:
            if (keyword in list_of_words_in_tweet):
                filterd_tweets.append(tweet)
    return filterd_tweets


def delete_other_than_persian_for_list_of_just_tweets(data):
    new_data = []
    for i in range(len(data)):
        temp = data[i].split()
        new_tweet_list = []
        for j in range(len(temp)):
            if (temp[j][0] not in lower_case and temp[j][0] not in upper_case):
                new_tweet_list.append(temp[j])
        new_tweet = ' '.join(new_tweet_list)
        new_data.append(new_tweet)
    return new_data


import string

lower_case = list(string.ascii_lowercase)
upper_case = list(string.ascii_uppercase)
from textblob import TextBlob


def delete_other_than_persian(data):
    new_data = []
    for i in range(len(data)):
        temp = data[i][0].split()
        new_tweet_list = []
        for j in range(len(temp)):
            if (temp[j][0] not in lower_case and temp[j][0] not in upper_case):
                new_tweet_list.append(temp[j])
        new_tweet = ' '.join(new_tweet_list)
        new_data.append([new_tweet, data[i][1]])
    return new_data


def translate2english_then_sentiment(text_persian):
    text_persian = delete_at_signs_for_list_of_just_tweets([text_persian])[0]
    text_persian = delete_other_than_persian_for_list_of_just_tweets([text_persian])[0]
    text_persian = delete_punctuation_marks(text_persian)
    translated_to_english = translate_persian2english([text_persian])
    # print(text_persian)
    # print(translated_to_english[0])
    sentiment = sample_analyze_sentiment(translated_to_english[0])
    # if(sentiment<-0.25):
    #   print('negative')
    #   return 0
    # elif(sentiment>-0.25 and sentiment<0.25):
    #   print('neutral')
    # elif(sentiment>0.25):
    #   print('positive')
    #   return 0
    # print(sentiment)
    if (sentiment < 0):
        return 0

    else:

        return 1


translate_persian2english("")

translate2english_then_sentiment()


def accuracy(x, y):
    correct = 0
    for i in range(len(x)):
        if (translate2english_then_sentiment(x[i]) == y[i]):
            correct += 1
    print("accuracy: ", correct / len(x))



accuracy(x, y)