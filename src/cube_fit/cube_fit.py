from cube_fit.linalg import *

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
    cube_vectors = nearest_orthogonal_matrix(*intersection_lines)

    # construct a transformation matrix from the origin points and basis vectors
    frame_transform = np.hstack((cube_vectors, corner_point.reshape(-1, 1)))
    frame_transform = np.vstack((frame_transform, np.array([0, 0, 0, 1])))

    return frame_transform

def six_point_cube_transform(face_points_3: np.ndarray, face_points_2: np.ndarray, face_points_1: np.ndarray):
    '''
    face_points_3 should contain 3 points in the first plane, 2 for face_points_2 and 1 for face_points_1
    From face_points_3, we can get the first plane.
    From face_points_2 and a projection of a point from face_points_2 on the first plane, we can get the second plane
    From face_points_1, we can get two projections on the previous 2 planes to get the last plane
    The planes are already othorgonal. Simply arrange them in [x.T,y.T,z.T] to get the frame_transform
    '''
    center_3 , normal_3 = plane_of_best_fit(face_points_3)
    face_points_2 = np.vstack([face_points_2, proj_point_on_plane(face_points_2[0], normal_3, center_3)])
    center_2, normal_2 = plane_of_best_fit(face_points_2)
    face_points_1 = np.vstack([face_points_1, proj_point_on_plane(face_points_1[0], normal_3, center_3)])
    face_points_1 = np.vstack([face_points_1, proj_point_on_plane(face_points_1[0], normal_2, center_2)])
    return find_cube_transform(face_points_3, face_points_2, face_points_1)
