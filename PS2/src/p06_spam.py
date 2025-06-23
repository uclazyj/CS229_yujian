import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split()
    return [word.lower() for word in words]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    occurrence_in_messages = collections.defaultdict(int)
    for message in messages:
        words = get_words(message)
        words_set = set(words)
        for word in words_set:
            occurrence_in_messages[word] += 1
    word_to_index = {}
    idx = 0
    for word in occurrence_in_messages:
        if occurrence_in_messages[word] >= 5:
            word_to_index[word] = idx
            idx += 1
    return word_to_index
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    m = len(messages)
    n = len(word_dictionary)
    a = np.zeros((m, n))
    for i, message in enumerate(messages):
        words = get_words(message)
        word_count = collections.Counter(words)
        for word in word_count:
            if word in word_dictionary:
                idx = word_dictionary[word]
                a[i][idx] = word_count[word]
    return a
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # The matrix is the output of transform_text. So matrix[i][j] represents the number of times the j_th word in the dictionary appear in message i.
    # We want a matrix contains 0 and 1 only, where matrix[i][j] = 1 means the j_th word in the dictionary appears in message i, and matrix[i][j] = 0 means it does not appear.
    X = matrix > 0
    m, n = X.shape
    spam = X[labels==1]
    nonspam = X[labels==0]
    phi_y = spam.shape[0] / m
    phi_y0 = (1 + nonspam.sum(axis=0)) / (2 + nonspam.shape[0])
    phi_y1 = (1 + spam.sum(axis=0)) / (2 + spam.shape[0])
    return (phi_y0, phi_y1, phi_y)
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_y0, phi_y1, phi_y = model
    X = matrix > 0

    # m, n = X.shape
    # predictions = []
    # for i in range(m):
    #     x = X[i]
    #     # When x_j = 1, P(x_j|y=1) = phi_j|y=1. When x_j = 0, P(x_j|y=1) = 1 - phi_j|y=1
    #     Px_y1 = phi_y1 * x + (1 - phi_y1) * (~x)
    #     Px_y0 = phi_y0 * x + (1 - phi_y0) * (~x)
    #     # Calculating the product directly will cause underflow.
    #     # Instead of calculating P1 * P2 * ... Pn, we calculate exp[ln(P1)+ln(P2)+...+ln(Pn)]
    #     Px_y1_product = np.exp(np.log(Px_y1).sum())
    #     Px_y0_product = np.exp(np.log(Px_y0).sum())
    #     P_y1 = (Px_y1_product * phi_y) / (Px_y1_product * phi_y + Px_y0_product * (1 - phi_y))
    #     predictions.append(P_y1)
    # return np.array(predictions) > 0.5

    Px_y1 = phi_y1 * X + (1 - phi_y1) * (~X)
    Px_y0 = phi_y0 * X + (1 - phi_y0) * (~X)
    Px_y1_product = np.exp(np.log(Px_y1).sum(axis=1))
    Px_y0_product = np.exp(np.log(Px_y0).sum(axis=1))
    P_y1 = (Px_y1_product * phi_y) / (Px_y1_product * phi_y + Px_y0_product * (1 - phi_y))
    return P_y1 > 0.5
    

    

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_y0, phi_y1, phi_y = model
    spam_metric = np.log(phi_y1 / phi_y0)
    print(spam_metric.shape)
    words = [(metric, i) for i, metric in enumerate(spam_metric)]
    words.sort(reverse = True)
    top_five_index = [words[i][1] for i in range(5)]
    index_to_word = {dictionary[word]: word for word in dictionary}
    top_five_words = [index_to_word[idx] for idx in top_five_index]
    return top_five_words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    a = []
    for radius in radius_to_consider:
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = (predictions == val_labels).mean()
        a.append((accuracy, radius))
    a.sort(reverse = True)
    return a[0][1]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
