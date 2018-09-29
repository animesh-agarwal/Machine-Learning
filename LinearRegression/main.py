import numpy as np
from LinearRegression.LinearModel import LinearRegressionUsingGD
from LinearRegression.Metrics import PerformanceMetrics
from LinearRegression.Plots import scatter_plot, plot, ploty


def generate_data_set():
    """ Generates Random Data

    Returns
    -------
    x : array-like, shape = [n_samples, n_features]
            Training samples
    y : array-like, shape = [n_samples, n_target_values]
            Target values

    """
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)
    return x, y


if __name__ == "__main__":
    # initializing the model
    linear_regression_model = LinearRegressionUsingGD()

    # generate the data set
    x, y = generate_data_set()

    # transform the feature vectors to include the bias term
    # adding 1 to all the instances of the training set.
    m = x.shape[0]
    x_train = np.c_[np.ones((m, 1)), x]

    # fit/train the model
    linear_regression_model.fit(x_train, y)

    # predict values
    predicted_values = linear_regression_model.predict(x_train)

    # model parameters
    print(linear_regression_model.w_)
    intercept, coeffs = linear_regression_model.w_

    # cost_function
    cost_function = linear_regression_model.cost_

    # plotting
    scatter_plot(x, y)
    plot(x, predicted_values)
    ploty(cost_function, 'no of iterations', 'cost function')

    # computing metrics
    metrics = PerformanceMetrics(y, predicted_values)
    rmse = metrics.compute_rmse()
    r2_score = metrics.compute_r2_score()

    print('The coefficient is {}'.format(coeffs))
    print('The intercept is {}'.format(intercept))
    print('Root mean squared error of the model is {}.'.format(rmse))
    print('R-squared score is {}.'.format(r2_score))
