import matplotlib.pyplot as plt

from linalg import *
from cube_fit.cube_fit import find_cube_transform


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


def plot_line(point_on_line, direction_vector, ax=None, length=1):
    ax = generate_axes(ax)

    # Calculate points on the line using the given point and direction vector
    line_points = np.array([point_on_line + t * direction_vector for t in np.linspace(-length, length, 50)])

    # Plot the line on top of the existing plot
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='b', label='Line')

    return ax


def plot_cube_fit(face_points_1: np.ndarray, face_points_2: np.ndarray, face_points_3: np.ndarray):
    ax = generate_axes()

    all_points = np.vstack((face_points_1, face_points_2, face_points_3))
    lower_bounds = np.min(all_points, axis=0)
    upper_bounds = np.max(all_points, axis=0)

    ax.set_xlim([lower_bounds[0], upper_bounds[0]])
    ax.set_ylim([lower_bounds[1], upper_bounds[1]])
    ax.set_zlim([lower_bounds[2], upper_bounds[2]])

    # TODO: loop over each group of face point to avoid code duplication
    ax.scatter(face_points_1[:, 0], face_points_1[:, 1], face_points_1[:, 2], c='r', marker='o')
    face_1_center, face_1_plane_normal = plane_of_best_fit(face_points_1)
    coeff = plane_coefficients(face_1_center, face_1_plane_normal)
    plot_plane_of_best_fit(face_points_1, coeff, ax)

    ax.scatter(face_points_2[:, 0], face_points_2[:, 1], face_points_2[:, 2], c='g', marker='o')
    face_2_center, face_2_plane_normal = plane_of_best_fit(face_points_2)
    coeff = plane_coefficients(face_2_center, face_2_plane_normal)
    plot_plane_of_best_fit(face_points_2, coeff, ax)

    ax.scatter(face_points_3[:, 0], face_points_3[:, 1], face_points_3[:, 2], c='b', marker='o')
    face_3_center, face_3_plane_normal = plane_of_best_fit(face_points_3)
    coeff = plane_coefficients(face_3_center, face_3_plane_normal)
    plot_plane_of_best_fit(face_points_3, coeff, ax)

    # fit lines of intersections between each plane y, z -> x
    point_on_line_x, direction_vector_x = line_of_intersection_2(face_3_center, face_3_plane_normal, face_1_center,
                                                                 face_1_plane_normal,
                                                                 0.5 * (face_3_center + face_1_center))
    plot_line(point_on_line_x, direction_vector_x, ax)
    ax.scatter(point_on_line_x[0], point_on_line_x[1], point_on_line_x[2], c='k', marker='o')

    point_on_line_y, direction_vector_y = line_of_intersection_2(face_1_center, face_1_plane_normal, face_2_center,
                                                                 face_2_plane_normal,
                                                                 0.5 * (face_1_center + face_2_center))
    plot_line(point_on_line_y, direction_vector_y, ax)
    ax.scatter(point_on_line_y[0], point_on_line_y[1], point_on_line_y[2], c='k', marker='o')

    point_on_line_z, direction_vector_z = line_of_intersection_2(face_2_center, face_2_plane_normal, face_3_center,
                                                                 face_3_plane_normal,
                                                                 0.5 * (face_2_center + face_3_center))
    plot_line(point_on_line_z, direction_vector_z, ax)
    ax.scatter(point_on_line_z[0], point_on_line_z[1], point_on_line_z[2], c='k', marker='o')

    # find three pair-wise points of intersections between the lines
    p1 = line_line_intersection(point_on_line_x, direction_vector_x, point_on_line_y, direction_vector_y)
    p2 = line_line_intersection(point_on_line_y, direction_vector_y, point_on_line_z, direction_vector_z)
    p3 = line_line_intersection(point_on_line_z, direction_vector_z, point_on_line_x, direction_vector_x)

    corner_point = np.mean(np.array([p1, p2, p3]), axis=0)

    corner_point_2 = find_cube_transform(face_points_1, face_points_2, face_points_3)

    ax.scatter(corner_point[0], corner_point[1], corner_point[2], c='k', marker='o', s=100)
    ax.scatter(corner_point_2[0], corner_point_2[1], corner_point_2[2], c='r', marker='o', s=50)

    plt.show()
