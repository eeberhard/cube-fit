from linalg import *


def find_cube_transform(face_points_1: np.ndarray, face_points_2: np.ndarray, face_points_3: np.ndarray):
    faces = [face_points_1, face_points_2, face_points_3]
    centers = []
    normals = []
    intersection_points = []
    intersection_lines = []

    # fit planes to faces
    for face in range(3):
        center, normal = plane_of_best_fit(faces[face])
        centers.append(center)
        normals.append(normal)

    # find intersection lines between each plane pair
    for i in range(3):
        j = i + 1 if i < 2 else 0
        point, line = \
            line_of_intersection_2(centers[i], normals[i], centers[j], normals[j], 0.5 * (centers[i] + centers[j]))
        intersection_points.append(point)
        intersection_lines.append(line)

    # find the intersection points between each line pair
    corner_points = []
    for i in range(3):
        j = i + 1 if i < 2 else 0
        corner_points.append(
            line_line_intersection(intersection_points[i], intersection_lines[i], intersection_points[j],
                                   intersection_lines[j]))

    corner_point = np.mean(np.array([corner_points[0], corner_points[1], corner_points[2]]), axis=0)

    # iteratively adjust the direction vectors to form an orthonormal basis
    # TODO

    # construct a transformation matrix from the origin points and basis vectors
    # TODO

    return corner_point
