import numpy as np
from scipy.spatial.transform import Rotation


def plane_of_best_fit_c(points: np.ndarray):
    """
    Find the plane of best fit for three or more 3D points

    :param points: Nx3 numpy array of points
    :return: The coefficients (a, b, c, d) of the plane equation ax + by + cz = d
    """
    if points.shape[0] < 3:
        raise RuntimeError('At least three points are required to fit a plane')
    if points.shape[1] != 3:
        raise RuntimeError('The points must be arranged as shape Nx3')

    # Extract the x, y, and z coordinates from the points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Build the design matrix A
    A = np.column_stack((x, y, np.ones_like(x)))

    # Compute the coefficients (a, b, c) of the plane equation Ax + By + Cz = D
    coefficients, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    a, b, c = coefficients

    # find the final coefficient
    d = -np.mean(a * x + b * y + c * z)

    return a, b, c, d


def plane_of_best_fit(points):
    """
    Find the plane of best fit for three or more 3D points

    :param points: Nx3 numpy array of points
    :return: The centroid of the points and the normal vector of the plane
    """
    if points.shape[0] < 3:
        raise RuntimeError('At least three points are required to fit a plane')
    if points.shape[1] != 3:
        raise RuntimeError('The points must be arranged as shape Nx3')

    # Calculate the centroid (average center point) of the points
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered_points = points - centroid

    # Perform PCA to find the normal vector of the plane (associated with smallest eigenvalue)
    _, _, V = np.linalg.svd(centered_points)
    normal_vector = V[-1]

    return centroid, normal_vector


def plane_coefficients(point_on_plane, normal_vector):
    a, b, c = normal_vector[0], normal_vector[1], normal_vector[2]
    d = np.dot(normal_vector, point_on_plane)
    return a, b, c, d


def line_of_intersection(plane_1, plane_2, preferred_point_on_line=None):
    """
    Find the line of intersection between two planes
    :param plane_1: Tuple of plane coefficients (a, b, c) of the plane equation Ax + By + Cz = D
    :param plane_2: Tuple of plane coefficients (a, b, c) of the plane equation Ax + By + Cz = D
    :return: The line of intersection represented by a point on the line and a direction vector
    """
    # Extract coefficients of the first plane
    a1, b1, c1, d1 = plane_1

    # Extract coefficients of the second plane
    a2, b2, c2, d2 = plane_2

    if preferred_point_on_line is None:
        preferred_point_on_line = [0, 0, 0]

    # Calculate the direction vector of the line of intersection
    direction_vector = np.cross([a1, b1, -1], [a2, b2, -1])
    direction_vector /= np.linalg.norm(direction_vector)
    print(direction_vector)

    # if the direction of the line tends along the Z axis, set X and solve for Y and Z
    if np.abs(np.dot(direction_vector, [1, 0, 0])) > 0.5:
        print("Setting X preferred")
        x = preferred_point_on_line[0]
        A = np.array([[b1, c1], [b2, c2]])
        b = np.array([-d1 - a1 * x, -d2 - a2 * x])
        intersection_point_yz = np.linalg.solve(A, b)
        intersection_point = np.append([x], intersection_point_yz)
    elif np.abs(np.dot(direction_vector, [0, 1, 0])) > 0.5:
        print("Setting Y preferred")
        y = preferred_point_on_line[1]
        A = np.array([[c1, a1], [c2, a2]])
        b = np.array([-d1 - b1 * y, -d2 - b2 * y])
        intersection_point_zx = np.linalg.solve(A, b)
        intersection_point = np.array([intersection_point_zx[1], y, intersection_point_zx[0]])
    else:
        print("Setting Z preferred")
        z = preferred_point_on_line[2]
        A = np.array([[a1, b1], [a2, b2]])
        b = np.array([-d1 - c1 * z, -d2 - c2 * z])
        intersection_point_xy = np.linalg.solve(A, b)
        intersection_point = np.append(intersection_point_xy, [z])

    return intersection_point, direction_vector


def line_of_intersection_2(p1, n1, p2, n2, p0=None):
    """
    """
    if p0 is None:
        p0 = np.array([0, 0, 0])

    M = np.array([[2, 0, 0, n1[0], n2[0]],
                  [0, 2, 0, n1[1], n2[1]],
                  [0, 0, 2, n1[2], n2[2]],
                  [n1[0], n1[1], n1[2], 0, 0],
                  [n2[0], n2[1], n2[2], 0, 0]])

    b4 = p1[0] * n1[0] + p1[1] * n1[1] + p1[2] * n1[2]
    b5 = p2[0] * n2[0] + p2[1] * n2[1] + p2[2] * n2[2]
    b = np.array([[2 * p0[0]],
                  [2 * p0[1]],
                  [2 * p0[2]],
                  [b4],
                  [b5]])

    # x = M\b;
    x = np.linalg.solve(M, b).T
    p = x[0, 0:3]
    n = np.cross(n1, n2)

    return p, n


def line_line_intersection(line1_point, line1_direction, line2_point, line2_direction):
    # Convert input points and direction vectors to NumPy arrays for easier manipulation
    line1_point = np.array(line1_point)
    line1_direction = np.array(line1_direction)
    line2_point = np.array(line2_point)
    line2_direction = np.array(line2_direction)

    # Build the matrix A for the system of equations Ax = b
    A = np.column_stack((line1_direction, -line2_direction))

    # Compute the vector b for the system of equations Ax = b
    b = line2_point - line1_point

    # Solve the system of equations to find the parameters t and s
    parameters, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Extract the parameters t and s
    t, s = parameters

    # Calculate the point of intersection or the point halfway between the lines
    intersection_point = line1_point + t * line1_direction

    # If the lines are parallel, the intersection_point will be the midpoint between the lines
    if np.isclose(t, s):
        midpoint = 0.5 * (line1_point + line2_point)
        return midpoint

    return intersection_point


def nearest_orthogonal_matrix(l1, l2, l3):
    M = np.vstack((l1,l2,l3)).T
    U, S, V = np.linalg.svd(M)
    R = U@V
    return R


def proj_point_on_plane(p0, n1, p1):
    '''
    p0 is a point to project on the plane
    n1 is the plane's normal
    p1 is a point on the plane
    '''
    p1p0 = p0 - p1
    projp0 = p0 - (np.dot(p1p0, n1) / np.dot(n1, n1)) * n1
    return projp0


def random_point_on_plane(n1, p1):
    distance_limit = 1
    random_coords = np.random.randn(3)
    random_direction = random_coords - np.dot(random_coords, n1) / np.dot(n1, n1) * n1
    random_direction /= np.linalg.norm(random_direction)
    random_distance = np.random.uniform(0, distance_limit)
    random_point = p1 + random_distance * random_direction
    return random_point


def distance_to_plane(p0, n1, p1):
    return np.abs(np.dot(p0 - p1, n1) / np.linalg.norm(n1))


def se3_difference(matrix1, matrix2):
    # Extract rotational and translational components from the SE3 matrices
    rot1 = Rotation.from_matrix(matrix1[:3, :3])
    trans1 = matrix1[:3, 3]

    rot2 = Rotation.from_matrix(matrix2[:3, :3])
    trans2 = matrix2[:3, 3]

    # Compute the differences
    rot_diff = rot1.inv() * rot2
    trans_diff = trans2 - trans1

    angle = rot_diff.as_rotvec()

    return trans_diff, np.linalg.norm(angle)np.linalg.norm(angle), 