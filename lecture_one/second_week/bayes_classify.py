from utils import process_tweet, lookup
from nltk.corpus import twitter_samples
import numpy as np
import nltk


# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = "{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)


def data_preprocess():
    # get the sets of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg
    # avoid assumptions about the length of all_positive_tweets
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
    test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))
    return train_x, train_y, test_x, test_y


def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1
            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    return result


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0
    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    # Calculate D, the number of documents
    D = len(train_y)
    # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    D_pos = np.sum(train_y)
    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = D - D_pos
    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)
        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)
    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p:the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
    '''
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)
    # initialize probability to zero
    p = 0
    # add the logprior
    p += logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]
    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0
        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)
    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.sum(np.abs(test_y - y_hats))/len(test_x)
    # Accuracy is 1 minus the error
    accuracy = 1 - error
    return accuracy


def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words
        word: string to lookup

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)
    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)
    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio


def get_words_by_threshold(freqs, label, threshold):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_set: dictionary containing the word and information on its positive count, negative count,
        and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}
    for key in freqs.keys():
        word, _ = key
        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)
        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio
        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio
        # otherwise, do not include this word in the list (do nothing)
    return word_list


def erro_analysis(test_x, test_y, logprior, loglikelihood):
    # Some error analysis done for you
    filename = 'error_analys.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        print("Files %s is recording the wrong tweet!" % filename)
        file_object.write("Add wrong tweet:\n")
        for x, y in zip(test_x, test_y):
            y_hat = naive_bayes_predict(x, logprior, loglikelihood)
            if y != (np.sign(y_hat) > 0):
                file_object.write('%d\t%0.2f\t%s\n' % (y, np.sign(y_hat) > 0,
                                                     ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
    print("Files %s have finished!" % filename)


def test_yourself_tweet(my_tweet, logprior, loglikelihood):
    # Test with your own tweet - feel free to modify `my_tweet`
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    return p


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = data_preprocess()
    freqs = count_tweets({}, train_x, train_y)
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    print("Naive Bayes accuracy = %0.4f" % (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))
    # Test your function: find negative words at or below a threshold
    get_words_by_threshold(freqs, label=0, threshold=0.05)
    # Test your function; find positive words at or above a threshold
    get_words_by_threshold(freqs, label=1, threshold=10)
    erro_analysis(test_x, test_y, logprior, loglikelihood)
    print(get_ratio(freqs, 'happi'))