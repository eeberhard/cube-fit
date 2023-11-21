import numpy as np
import matplotlib.pyplot as plt


def generate_axes(ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    return ax


def plot_plane_of_best_fit(points: np.ndarray, plane: tuple, ax=None):
    # Plot the points and the plane of best fit
    ax = generate_axes(ax)

    # Extract plane coefficients
    a, b, c, d = plane

    # Create a meshgrid of points for the plane
    xrange = [min([p[0] for p in points]), max([p[0] for p in points])]
    yrange = [min([p[1] for p in points]), max([p[1] for p in points])]
    zrange = [min([p[2] for p in points]), max([p[2] for p in points])]

    smallest_range = np.argmin([xrange[1] - xrange[0], yrange[1] - yrange[0], zrange[1] - zrange[0]])
    if smallest_range == 0:
        # solve for X
        Y, Z = np.meshgrid(yrange, zrange)
        X = (d - b * Y - c * Z) / a
    elif smallest_range == 1:
        # solve for Y
        Z, X = np.meshgrid(zrange, xrange)
        Y = (d - a * X - c * Z) / b
    else:
        # solve for Z
        X, Y = np.meshgrid(xrange, yrange)
        Z = (d - a * X - b * Y) / c

    # Scatter plot for the initial points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Data Points')

    # Surface plot for the plane of best fit
    ax.plot_surface(X, Y, Z, alpha=0.5, label='Plane of Best Fit')

    return ax


def plot_line(point_on_line, direction_vector, ax=None, length=1, color='red'):
    ax = generate_axes(ax)

    # Calculate points on the line using the given point and direction vector
    line_points = np.array([point_on_line + t * direction_vector for t in np.linspace(-length, length, 50)])

    # Plot the line on top of the existing plot
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c=color, label='Line')

    return ax
