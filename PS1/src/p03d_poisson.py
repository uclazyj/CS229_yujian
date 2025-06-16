import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    
    m,n = x_train.shape
    theta_0 = np.zeros(n)
    clf = PoissonRegression(step_size=lr, max_iter=1000, eps=1e-5, theta_0=theta_0)
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_valid)
    util.save_prediction(y_pred, pred_path, usePandas=False)

    plt.plot(y_valid, y_pred, 'bx')
    plt.plot(y_valid, y_valid, 'r', label='y=x')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.legend()
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        
        theta_old = self.theta
        for it in range(self.max_iter):
            h = np.exp(x @ theta_old)
            self.theta = theta_old + self.step_size * x.T @ (y - h)
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
            theta_old = self.theta
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
