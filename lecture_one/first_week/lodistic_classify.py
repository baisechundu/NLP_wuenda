import numpy as np
from nltk.corpus import twitter_samples
from nlputils import process_tweet, build_freqs


def data_process():
    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg
    # combine positive and negative labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
    return train_x, train_y, test_x, test_y


def sigmoid(z):
    '''
    Input:  z: is the input (can be a scalar or an array)
    Output: h: the sigmoid of z
    '''
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    return h


def gradientdescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    # get 'm', the number of rows in matrix x
    m = x.shape[0]
    J = 0.0
    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)
        # get the sigmoid of z
        h = sigmoid(z)
        # calculate the cost function
        J = -(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) / m
        # update the weights theta
        theta = theta - alpha * (np.dot(x.T, (h - y))) / m
    J = float(J)
    return J, theta


def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))
    # bias term is set to 1
    x[0, 0] = 1
    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)
    assert (x.shape == (1, 3))
    return x


def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    # the list for storing predictions
    y_hat = []
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0)
    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = np.sum((np.array(y_hat) == test_y.flatten()) != 0) / len(y_hat)
    return accuracy


def test_self_tweet(my_tweet, freq, the):
    print(process_tweet(my_tweet))
    y_hat = predict_tweet(my_tweet, freq, the)
    print(y_hat)
    if y_hat > 0.5:
        print('Positive sentiment')
    else:
        print('Negative sentiment')


def main():  # 模型的训练以及错分样本的记录输出
    train_x, train_y, test_x, test_y = data_process()
    # create frequency dictionary
    freqs_train = build_freqs(train_x, train_y)
    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs_train)
    # training labels corresponding to X
    Y = train_y
    # Apply gradient descent
    m, omega = gradientdescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
    print("The cost after training is %.8f" % m)
    tmp_accuracy = test_logistic_regression(test_x, test_y, freqs_train, omega)
    print("Logistic regression model's accuracy = %.4f" % tmp_accuracy)
    # Some error analysis done for you
    filename = 'mis_classify.txt'
    with open(filename, 'w', encoding='utf-8') as file_object:
        print("Files mis_classify is recording the wrong tweet!")
        file_object.write("Add a word\n")
        for x, y in zip(test_x, test_y):
            y_hat = predict_tweet(x, freqs_train, omega)
            if np.abs(y - (y_hat > 0.5)) > 0:
                file_object.write('THE TWEET IS: %s \n' % x)
            file_object.write('THE PROCESSED TWEET IS: %s \n' % process_tweet(x))
            file_object.write('%d\t%0.8f\t%s \n' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
    print('Files mis_classify is finished.')
    return freqs_train, omega


if __name__ == '__main__':
    frequences, theta = main()
    test_sentence = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
    test_self_tweet(test_sentence, freq=frequences, the=theta)