import tweepy
import configs
from datetime import datetime
import sentiment
import mongo_dbi

consumer_key= configs.consumer_key
consumer_secret= configs.consumer_secret
access_key = configs.access_key
access_secret = configs.access_secret

# function to perform data extraction
def fetch(words, date_since, numtweet):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    tweets = tweepy.Cursor(api.search_tweets, words, lang="en", since_id=date_since, tweet_mode='extended').items(numtweet)

    list_tweets = [tweet for tweet in tweets]

    # we will iterate over each tweet in the list for extracting information about each tweet
    for tweet in list_tweets:
        #print(tweet)
        
        datax = {"username" : tweet.user.screen_name}
        datax["_id"] = tweet.id
        datax["description"] = tweet.user.description
        datax["location"] = tweet.user.location
        datax["tweet_date"] = tweet.created_at.date()
        
        #tweet_date = datetime.strptime(tweet_date, '%a %b %d %H:%M:%S %z %Y').date()
        #print(tweet_date)

        # Retweets can be distinguished by a retweeted_status attribute
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        datax["text"] = text
        datax["sentiment"] = sentiment.get_sentiment(text)
        
        try:
            media_url = tweet.entities["media"][0]["media_url"]
        except:
            media_url = ""
        datax["media_url"] = media_url
        inserted = mongo_dbi.insert_tweet_data(datax)
        print(inserted)

def get_tweets(keyword, start_date, numtweet):
    words = keyword
    date_since = start_date #"2022-04-19"
    fetch(words, date_since, numtweet)
    print('Scraping has completed!')

#get_tweets("Tesla","2022-04-19", 1)

"""
#Streaming
class IDPrinter(tweepy.Stream):

    def on_status(self, status):
        #print(status.entities)
        #print(status.text)
        try:
            media_url = status.entities["media"][0]["media_url"]
            print(media_url)
        except Exception as e:
            pass

streamer = IDPrinter(consumer_key, consumer_secret, access_key, access_secret)
streamer.filter(track=[keyword])
#printer.sample()
"""