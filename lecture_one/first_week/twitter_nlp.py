import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nlputils import process_tweet
import os


def all_data():
    if os.path.exists(r'C:\Users\baiyang01\AppData\Roaming\nltk_data\corpora\twitter_samples'):
        print('Files already exists.')
        pass
    else:
        nltk.download('twitter_samples')  # select the set of positive and negative tweets
        print("I'm downloading the file")
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    return positive_tweets, negative_tweets


def plot_data(positive_tweets, negative_tweets):
    plt.figure(figsize=(5, 5))
    labels = 'Positives', 'Negative'  # labels for the two classes
    sizes = [len(positive_tweets), len(negative_tweets)]  # Sizes for each slide
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)  # Declare pie chart
    # where the slices will be ordered and plotted counter-clockwise:
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()  # Display the chart


def tweet_split(tweet):
    if os.path.exists(r'C:\Users\baiyang01\AppData\Roaming\nltk_data\corpora\stopwords'):
        print('Files already exists.')
        pass
    else:
        nltk.download('stopwords')  # download the stopwords from NLTK
        print("I'm downloading the file")
    print('\033[92m' + tweet)
    print('\033[94m')
    tweet2 = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)   # remove hyperlinks
    tweet2 = re.sub(r'#', '', tweet2)  # only removing the hash # sign from the word
    print(tweet2)
    print()
    print('\033[92m' + tweet2)
    print('\033[94m')
    # instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                   reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet2)  # tokenize tweets
    print('Tokenized string:')
    print(tweet_tokens)
    stopwords_english = stopwords.words('english')  # Import the english stop words list from NLTK
    print('Stop words\n')
    print(stopwords_english)
    print('\nPunctuation\n')
    print(string.punctuation)
    print()
    print('\033[92m')
    print(tweet_tokens)
    print('\033[94m')
    tweets_clean = []
    for word in tweet_tokens:  # Go through every word in your tokens list
        if (word not in stopwords_english and word not in string.punctuation):  # remove stopwords and punctuation
            tweets_clean.append(word)
    print('removed stop words and punctuation:')
    print(tweets_clean)
    print()
    print('\033[92m')
    print(tweets_clean)
    print('\033[94m')
    stemmer = PorterStemmer()  # Instantiate stemming class
    tweets_stem = []  # Create an empty list to store the stems
    for word in tweets_clean:
        stem_word = stemmer.stem(word)  # stemming word
        tweets_stem.append(stem_word)  # append to the list
    print('stemmed words:')
    print(tweets_stem)


def nlp_utils_processtweet(tweet):
    print('\033[92m')
    print(tweet)
    print('\033[94m')
    tweets_stem = process_tweet(tweet)  # Preprocess a given tweet
    print('preprocessed tweet:')
    print(tweets_stem)  # Print the result


def main():
    all_positive_tweets, all_negative_tweets = all_data()
    print('Number of positive tweets: ', len(all_positive_tweets))
    print('Number of negative tweets: ', len(all_negative_tweets))
    print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
    print('The type of a tweet entry is: ', type(all_negative_tweets[0]))
    plot_data(all_positive_tweets, all_negative_tweets)
    tweet = all_positive_tweets[2277]
    print('This is a self-defined text_split:\n')
    tweet_split(tweet)
    print('This is a utils_process_data:\n')
    nlp_utils_processtweet(tweet)


if __name__ == "__main__":
    main()
