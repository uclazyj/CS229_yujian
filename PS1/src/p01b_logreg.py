import numpy as np
import util
from random import random

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    # *** START CODE HERE ***

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    m,n = x_train.shape
    
    # Initializing theta_0 to a zero vector is crucial! Otherwise, we will run into the problem of an non-invertible Hessian matrix.
    
    theta_0 = np.zeros(n)

    clf = LogisticRegression(step_size=0.2, max_iter=100, eps=1e-5, theta_0=theta_0)
    clf.fit(x_train, y_train)
    util.plot(x_train, y_train, clf.theta, title='training set (logistic regression)')
    util.plot(x_eval, y_eval, clf.theta, title='validation set (logistic regression)')

    y_pred = clf.predict(x_eval)
    accuracy = util.accuracy_score(y_eval, y_pred)
    print(f"The accuracy of the logistic regression model is: {100 * accuracy:.1f} %")

    util.save_prediction(y_pred, pred_path, usePandas=True)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        m,n = x.shape
        for it in range(self.max_iter):
            g = self.sigmoid(x @ self.theta)
            H = (1/m) * x.T * g * (1 - g) @ x
            H_inv = np.linalg.inv(H)
            grad = (1/m) * x.T @ (g-y)
            theta_next = self.theta - H_inv @ grad
            if np.linalg.norm(theta_next - self.theta, ord=1) < self.eps:
                self.theta = theta_next
                break
            self.theta = theta_next
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        z = x @ self.theta
        self.h = self.sigmoid(z)
        y_pred = z >= 0.5
        return y_pred.astype(int)
        
        # *** END CODE HERE ***

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
