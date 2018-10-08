from sklearn.metrics import mean_squared_error, r2_score


class LinearModel:
    """ Trains a Linear Model model and computes the metrics

    Parameters
    ----------

    model : linear model on which the data will be trained

    Attributes
    ----------

    intercept_ : intercept of the linear line
    coeffs_ :coeffs of the model
    rmse_ : root mean squared error of the model
    r2_: r-squared error of the model


    """

    def __init__(self, model):
        self.model = model

    def compute_metrics(self, x, y):
        """Trains a model and computes the metrics

        Parameters
        ----------


        x : array-like, shape = [n_samples, n_features]
           Training samples
        y : array-like, shape = [n_samples, n_target_values]
          Target values

        Returns
        -------
        self: instance of self

        """

        self.model.fit(x, y)
        y_predicted = self.model.predict(x)
        self.intercept_ = self.model.intercept_
        self.coeffs_ = self.model.coeffs_
        self.rmse_ = mean_squared_error(y, y_predicted)
        self.r2_ = r2_score(y, y_predicted)
        return self
