import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    
    clf = GDA()
    clf.fit(x_train, y_train)

    util.plot(x_train, y_train, [clf.theta0] + list(clf.theta), title='training set (GDA)')
    util.plot(x_eval, y_eval, [clf.theta0] + list(clf.theta), title='validation set (GDA)')
    
    y_pred = clf.predict(x_eval)

    accuracy = util.accuracy_score(y_eval, y_pred)
    print(f"The accuracy of the GDA model is: {100 * accuracy:.1f} %")

    util.save_prediction(y_pred, pred_path, usePandas=True)    
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = np.sum(y) / m

        x0 = x[y==0]
        mu0 = np.mean(x0, axis=0)
        x1 = x[y==1]
        mu1 = np.mean(x1, axis=0)
        x0 = x0 - mu0
        x1 = x1 - mu1
        sigma = (x0.T @ x0 + x1.T @ x1) / m
        sigma_inv = np.linalg.inv(sigma)

        self.theta = sigma_inv @ (mu1 - mu0)
        self.theta0 = (mu0.T @ sigma_inv @ mu0 - mu1.T @ sigma_inv @ mu1)/2 + np.log(phi / (1 - phi))
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # p(y=1 | x)
        p = 1 / (1 + np.exp(-(x @ self.theta + self.theta0)))
        y_pred = p >= 0.5
        return y_pred.astype(int)
        # *** END CODE HERE
