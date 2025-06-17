import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data

    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    y_diff = y_pred - y_eval
    MSE = (y_diff * y_diff).mean()
    
    plt.plot(x_train[:,-1], y_train, 'bx', label = 'training data')
    plt.plot(x_eval[:,-1], y_pred, 'ro',label = 'predictions')
    plt.title(fr"$\tau$ = {tau} (MSE = {MSE:.3f})")
    plt.legend()
    plt.show()

    return MSE
    
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x_array):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x_array.shape # Note this m is not the same as the m of the training set
        # w is a m by 1 vector where w_i corresponds to W_ii in the formula
        y_pred = np.zeros(m)
        for i in range(m):
            x = x_array[i][-1]
            d = self.x[:,-1] - x
            w = np.exp(- d * d / 2 / self.tau ** 2)
            theta = np.linalg.inv(self.x.T @ (self.x * w.reshape(-1,1))) @ self.x.T @ (w * self.y)
            y_pred[i] = x_array[i,:] @ theta
        return y_pred
        
        # *** END CODE HERE ***
