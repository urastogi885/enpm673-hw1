import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def least_square_line_fit(x_data, y_data):
    """ find parameters to a parabola that fits the given data
    :param x_data: data on the independent axis
    :param y_data: corresponding dependent values for x
    :return: return a list of parameters to the parabola
    """
    # Generate matrix: [x^2 x 1] for parabola
    x_matrix = np.vstack([x_data ** 2, x_data, np.ones(len(x_data))]).T
    # Get and return a, b, c parameters for the parabola equation: y = ax^2 + bx + c
    # Use least squares solution: B = ((X^T.X)^(-1))(X^T.Y)
    return (np.linalg.inv(x_matrix.T.dot(x_matrix)).dot(x_matrix.T)).dot(y_data)


def ransac(data_x, data_y, max_iterations, threshold, inlier_ratio, n=3):
    """ custom method to run RANSAC on given dataset
    :param data_x: data on the independent axis
    :param data_y: corresponding dependent values for x
    :param max_iterations: maximum iterations for the method to run
    :param threshold: error threshold to fit data points
    :param inlier_ratio: minimum ratio of inliers required to fit the model
    :param n: minimum no. of points required to fit the model
    :return: a tuple of model parameters and total no. of fitting points
    """
    total_fitting_pts = 0
    best_model = None

    for _ in range(max_iterations):
        # Get a random sample of x points from the given dataset
        sample_inliers = random.sample(range(0, len(data_x)), n)
        # Instantiate lists to get new inliers apart from the sample points
        new_inliers_x = []
        new_inliers_y = []
        # Get model parameters w.r.t minimum no. of points required to fit the model
        params = least_square_line_fit(data_x[sample_inliers], data_y[sample_inliers])
        # Check and add points meeting the threshold requirements
        for j in range(len(data_x)):
            if abs(data_y[j] - (params[0]*(data_x[j]**2) + params[1]*data_x[j] + params[2])) < threshold:
                new_inliers_x.append(data_x[j])
                new_inliers_y.append(data_y[j])
        # Get ratio of inliers for the current sample
        current_ratio = (len(new_inliers_x)) / len(data_x)
        # Add model as best model if it surpasses previous inlier ratio
        if current_ratio >= inlier_ratio:
            # Update inlier ratio to only get better results
            inlier_ratio = current_ratio
            total_fitting_pts = len(new_inliers_x)
            # Re-fit the model with all the data points
            best_model = least_square_line_fit(np.asarray(new_inliers_x), np.asarray(new_inliers_y))

    return best_model, total_fitting_pts


if __name__ == '__main__':
    # Run for all given data-sets
    for i in range(1, 3):
        # Read the data from the csv file
        dataset = pd.read_csv('data/data_' + str(i) + '.csv')
        # Get x and y values from the dataset
        x = dataset.iloc[:, 0].values
        y = dataset.iloc[:, 1].values

        # Run least square line fitting method
        # Find parameters to fit the dataset using RANSAC
        model_params = least_square_line_fit(x, y)
        print(model_params)
        # Get all values for the fitted parabola model: y = ax^2 + bx + c
        model_values = model_params[0] * (x ** 2) + model_params[1] * x + model_params[2]
        # Plot the dataset and the model for least square method
        x_label = 'X-axis'
        y_label = 'Y-axis'
        plt.figure(i)
        plt.title('Fit using LS for Data ' + str(i))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.scatter(x, y, c='green', label='dataset')
        plt.plot(x, model_values, color='black', label='fitted curve')
        plt.legend()
        plt.show()

        # Define various parameters to run RANSAC
        iterations = 2000                       # Total no. of iterations
        t = 35 if i == 1 else 45                # error threshold
        inlier_prob = 0.98 if i == 1 else 0.85  # ratio of inliers
        # Find parameters to fit the dataset using RANSAC
        model_params = ransac(x, y, iterations, t, inlier_prob)
        print(model_params[0])
        # Get all values for the fitted parabola model: y = ax^2 + bx + c
        model_values = model_params[0][0]*(x**2) + model_params[0][1]*x + model_params[0][2]

        # Plot the dataset and the model for RANSAC method
        plt.figure(i+2)
        plt.title('Fit using RANSAC for Data ' + str(i))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.scatter(x, y, c='green', label='dataset')
        plt.plot(x, model_values, color='black', label='fitted curve')
        plt.plot(x, model_values + t, color='blue', linestyle='dashed', label='threshold')
        plt.plot(x, model_values - t, color='blue', linestyle='dashed')
        plt.legend()
        plt.show()

