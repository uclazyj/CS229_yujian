import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    m, n = x_train.shape
    theta_0 = np.zeros(n)

    clf_t = LogisticRegression(step_size=0.2, max_iter=100, eps=1e-4, theta_0=theta_0)
    clf_t.fit(x_train, t_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    predictions = clf_t.predict(x_test)
    util.save_prediction(predictions, pred_path_c)

    accuracy = util.accuracy_score(t_test, predictions)
    print(f"The accuracy of the logistic regression model is: {100 * accuracy:.1f} %")

    util.plot(x_train, t_train, clf_t.theta, title='training set (trained with t labels)')
    util.plot(x_test, t_test, clf_t.theta, title='test set (trained with t labels)')
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d


    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    m, n = x_train.shape
    theta_0 = np.zeros(n)

    clf_y = LogisticRegression(step_size=0.2, max_iter=100, eps=1e-5, theta_0=theta_0)
    clf_y.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    predictions = clf_y.predict(x_test)
    util.save_prediction(predictions, pred_path_d)

    accuracy = util.accuracy_score(y_test, predictions)
    print(f"The accuracy of the logistic regression model is: {100 * accuracy:.1f} %")

    util.plot(x_train, y_train, clf_y.theta, title='training set (trained with y labels)')
    util.plot(x_test, y_test, clf_y.theta, title='test set (trained with y labels)')

    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_valid_labeled = x_valid[y_valid==1]
    clf_y.predict(x_valid_labeled) # I don't need the predictions. I just need to run it to generate h(x)
    alpha_estimated = np.mean(clf_y.h)

    theta_corrected = clf_y.theta
    theta_corrected[0] += np.log(2/alpha_estimated - 1)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    util.plot(x_test, t_test, theta_corrected, title='test set (trained with y labels and apply correction)')

    pred = x_test @ theta_corrected > 0
    pred = pred.astype(int)
    util.save_prediction(predictions, pred_path_e)
    
    accuracy = util.accuracy_score(t_test, pred)
    print(f"The accuracy of the logistic regression model is: {100 * accuracy:.10f} %")

    # Sanity check
    print('The estimated alpha is:',alpha_estimated)

    def label_ratio(dataset_path):
        # Among all positive examples (t=1), what is the portion of the data is labeled (y=1)
        x, y = util.load_dataset(train_path, label_col='y', add_intercept=True)
        x, t = util.load_dataset(train_path, label_col='t', add_intercept=True)
        return (y==1).sum() / (t==1).sum()
    print('In the training set, among all the positive data (t=1), the ratio of the data labeled (y=1) is:',label_ratio(train_path))
    print('In the validation set, among all the positive data (t=1), the ratio of the data labeled (y=1) is:',label_ratio(valid_path))
    print('In the test set, among all the positive data (t=1), the ratio of the data labeled (y=1) is:',label_ratio(test_path))
    
    # *** END CODER HERE
