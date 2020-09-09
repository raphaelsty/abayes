from ..ar import ARModel

from scipy import stats

import numpy as np

__all__ = ['BayesLr', 'AbayesLr']


class BayesLr:
    """
    Bayesian linear regression assuming that noise follows Gaussian ansatz distribution.
    It implies that the likelihood function is a Gaussian distribution.

    Parameters:

        n_features (int): Number of features.
        alpha (float): The inverse of alpha is used to initialise the diagonal of the variance
            covariance matrix.
        beta (float): Noise precision parameter. It translates our belief on how noisy the target
            distribution is.

    Attributes:

        mean (np.ndarray): Mean of the distribution initialized to 0.
        cov_inv (np.ndarray): Covariance matrix.

    References:
        1. [Bayesian linear regression for practitioners - Max Halford](https://maxhalford.github.io/blog/bayesian-linear-regression-)

    Example:

        >>> from abayes import linear
        >>> import numpy as np

        >>> model = linear.BayesLr(
        ...     n_features = 3,
        ...     alpha = 0.3,
        ...     beta = 1,
        ... )

        >>> X_y = [
        ...     (np.array([[1, 2, 3]]), 2),
        ...     (np.array([[4, 5, 6]]), 5),
        ...     (np.array([[7, 8, 9]]), 8),
        ... ]

        >>> for x, y in X_y:
        ...     model.predict(x)
        ...     model = model.learn(x, y)
        0.0
        3.6923076923076925
        7.595898673100109

        >>> for x, _ in X_y:
        ...    model.predict_interval(x, alpha = 0.95)
        (-0.15876253102744586, 4.217555545034022)
        (2.7360668569206967, 7.177472828363761)
        (5.318687631175099, 10.449598725387242)

        >>> X = np.array([
        ...    [1, 2, 3],
        ...    [4, 5, 6],
        ...    [7, 8, 9],
        ... ])

        >>> y = np.array([2, 5, 8])

        >>> model = linear.BayesLr(
        ...     n_features = 3,
        ...     alpha = 0.3,
        ...     beta = 1,
        ... )

        >>> model = model.learn(X, y)

        >>> model.predict(X)
        array([2.02939651, 4.95676984, 7.88414318])

        >>> lower_bound, upper_bound = model.predict_interval(X, alpha = 0.95)

        >>> for l, u, t in zip(lower_bound, upper_bound, y):
        ...     print(f'[{l:6f}, {u:6f}], y: {t}')
        [-0.158763, 4.217556], y: 2
        [2.736067, 7.177473], y: 5
        [5.318688, 10.449599], y: 8

    """

    def __init__(self, n_features, alpha, beta):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros(n_features)
        self.cov_inv = np.identity(n_features) / alpha

    def learn(self, x, y):
        # If x and y are singletons, then we coerce them to a batch of length 1
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        # Update the inverse covariance matrix (Bishop eq. 3.51)
        cov_inv = self.cov_inv + self.beta * x.T @ x

        # Update the mean vector (Bishop eq. 3.50)
        cov = np.linalg.inv(cov_inv)
        mean = cov @ (self.cov_inv @ self.mean + self.beta * y @ x)

        self.cov_inv = cov_inv
        self.mean = mean

        return self

    def _predict(self, x):
        x = np.atleast_2d(x)
        # Obtain the predictive mean (Bishop eq. 3.58)
        y_pred_mean = x @ self.mean

        # Obtain the predictive variance (Bishop eq. 3.59)
        w_cov = np.linalg.inv(self.cov_inv)
        y_pred_var = 1 / self.beta + (x @ w_cov * x).sum(axis=1)

        # Drop a dimension from the mean and variance in case x and y were singletons
        # There might be a more elegant way to proceed but this works!
        y_pred_mean = np.squeeze(y_pred_mean)
        y_pred_var = np.squeeze(y_pred_var)

        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)

    def predict(self, x):
        """Returns the most likely value given the x entry."""
        return self._predict(x).mean()

    def predict_interval(self, x, alpha):
        """Returns the confidence interval with respect to alpha risk given the x entry."""
        return self._predict(x).interval(alpha)

    @property
    def weights_dist(self):
        """Model weights uncertainty."""
        cov = np.linalg.inv(self.cov_inv)
        return stats.multivariate_normal(mean=self.mean, cov=cov)


class AbayesLr(ARModel):
    """Auto regressive bayesian linear regression.

    Parameters:
        p (int): Period.
        alpha (float): The inverse of alpha is used to initialise the diagonal of the variance
            covariance matrix.
        beta (float): Noise precision parameter. It translates our belief on how noisy the target
            distribution is.

    Example:

        >>> from abayes import linear

        >>> model = linear.AbayesLr(
        ...    p     = 3,
        ...    alpha = 0.3,
        ...    beta  = 1.,
        ... )

        >>> X = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

        >>> model = model.learn(X)

        >>> model.forecast(6)
        array([1.26864728, 1.93952389, 2.57607348, 1.41884618, 1.8657895 ,
            2.26730957])

        Train the model 10 more times:
        >>> for _ in range(10):
        ...     model = model.learn(X)

        >>> model.forecast(6)
        array([1.02778671, 1.99451375, 2.96090061, 1.05446922, 1.98902856, 2.92293025])

        >>> lower_bound, upper_bound = model.forecast_interval(6, alpha = 0.95)

        >>> lower_bound
        array([-0.96186235,  0.00530674,  0.97171083, -0.93384501,  0.00112221,
            0.93504114])

        >>> upper_bound
        array([3.01743576, 3.98372076, 4.95009039, 3.04278345, 3.97693491,
           4.91081936])
    """

    def __init__(self, p, alpha, beta):

        self.model = BayesLr(
            n_features=p,
            alpha=alpha,
            beta=beta
        )

        super().__init__(
            p=p,
            model=self.model
        )

    @property
    def weights_dist(self):
        return self.model.weights_dist
