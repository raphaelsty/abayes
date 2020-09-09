
import numpy as np


__all__ = ['ARModel']


class NotEnoughDataToForecast(Exception):
    pass


class ARModel:
    """Auto regression class for online purpose.

    References:

        1. [Idao 2020 solution - Max Halford, Raphaël Sourty, Robin Vaysse](https://github.com/MaxHalford/idao-2020-qualifiers/blob/master/auto-regression.ipynb)

    """

    def __init__(self, p, model):
        self.p = p
        self.model = model
        self.history = np.empty(0)
        self.fitted = False

    def learn(self, path):

        length_path = len(path.flatten())

        # Extract historical data to make prediction:
        if length_path <= self.p:

            length_history = len(self.history.flatten())

            # If there are enough data available in history to make prediction:
            if (length_path + length_history) > self.p:

                path = np.concatenate(
                    (self.history.flatten(), path)[-self.p:],
                    axis=None,
                )

            # Not enough data to update the model for now.
            else:

                self.history = np.concatenate(
                    (self.history, path.reshape(1, -1)),
                    axis=None,
                )

                return self

        # When already fitted add last target value:
        elif self.fitted:
            path = np.insert(path, 0, self.history[-1])

        self.fitted = True

        n = path.strides[0]

        X = np.lib.stride_tricks.as_strided(
            path,
            shape=(path.shape[0], self.p),
            strides=(n, n)
        )[:-self.p]

        Y = path[self.p:].squeeze()

        # Save the most recent history for later usage
        self.history = path[-self.p:].reshape(1, -1)

        self.model.learn(X, Y)

        return self

    def forecast(self, steps):

        history = self.history.copy().astype(float)

        if len(history.flatten()) < self.p or not self.fitted:
            raise NotEnoughDataToForecast()

        predictions = np.empty(steps)

        for i in range(steps):

            y_pred = self.model.predict(history)

            predictions[i] = y_pred

            # Shift forward (faster than np.roll)
            history[0, :-1] = history[0, 1:]

            history[0, -1] = y_pred

        return predictions

    def forecast_interval(self, steps, alpha):

        history = self.history.copy().astype(float)

        if len(history.flatten()) < self.p or not self.fitted:
            raise NotEnoughDataToForecast()

        lower_bound = np.empty(steps)
        upper_bound = np.empty(steps)

        for i in range(steps):

            y_pred = self.model.predict(history)

            lower_bound[i], upper_bound[i] = self.model.predict_interval(
                x=history,
                alpha=alpha
            )

            # Shift forward (faster than np.roll)
            history[0, :-1] = history[0, 1:]

            history[0, -1] = y_pred

        return lower_bound, upper_bound
