# imports
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from PolynomialRegression.Models import LinearModel


def generate_data_set():
    np.random.seed(0)
    x = 2 - 3 * np.random.normal(0, 1, 20)
    y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x, y


if __name__ == "__main__":
    x_train, y_train = generate_data_set()

    # create a linear regression model and fit the data
    model = LinearRegression()
    linear_regression = LinearModel(model)
    linear_regression.compute_metrics(x_train, y_train)

    # printing metrics of the linear model
    print('The RMSE of the linear regression model is {}'.format(linear_regression.rmse_))
    print('The R2 score of the linear regression model is {}'.format(linear_regression.r2_))

    # transform the features to higher degree
    polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly_train = polynomial_features.fit_transform(x_train)

    # train a linear model with higher degree features
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    polynomial_regression = LinearModel(model)
    polynomial_regression.compute_metrics(x_poly_train, y_train)

    print('The RMSE of the polynomial regression model is {}'.format(polynomial_regression.rmse_))
    print('The R2 score of the polynomial regression model is {}'.format(polynomial_regression.r2_))
