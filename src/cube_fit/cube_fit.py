from cube_fit.linalg import *
from cube_fit.plot import *
from cube_fit.linalg import *
import state_representation as sr
from pyquaternion import Quaternion

def find_cube_transform(face_points_1: np.ndarray, face_points_2: np.ndarray, face_points_3: np.ndarray):
    # example points for testing
    # face_points_1 = np.array([
    #     [-0.61706884765625, 0.488536956787109, 0.0098738136291504],
    #       [-0.609881469726563, 0.495773071289063, 0.0097553081512451],
    #       [-0.624020568847656, 0.481449676513672, 0.0100793151855469],
    #       [-0.60986181640625, 0.481514465332031, 0.009827692985534701]
    # ])
    # face_points_2 = np.array([
    #     [-0.637412048339844, 0.505201934814453, 0.0007387455701828],
    #     [-0.6310705566406251, 0.513177612304688, -0.0013162076473236099],
    #     [-0.644199279785156, 0.49833343505859395, 0.0004966171681880949],
    #     [-0.63798046875, 0.505906646728516, -0.0106414880752563]
    # ])
    # face_points_3 = np.array([
    #     [-0.6006079711914061, 0.508934661865234, 0.0006297954320907591],
    #     [-0.607918273925781, 0.5158516845703129, 0.000651176035404205],
    #     [-0.600609985351563, 0.5086584777832031, -0.0094649076461792],
    #     [-0.593725769042969, 0.501691833496094, 0.0008056240677833561]
    # ])
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
    # print(frame_transform)
    ax = generate_axes()
    ax.set_xlim([-0.65, -0.55])
    ax.set_ylim([0.45, 0.55])
    ax.set_zlim([-0.05, 0.05])
    ax.scatter(face_points_1[:, 0], face_points_1[:, 1], face_points_1[:, 2], c='r', marker='o')
    ax.scatter(face_points_2[:, 0], face_points_2[:, 1], face_points_2[:, 2], c='g', marker='o')
    ax.scatter(face_points_3[:, 0], face_points_3[:, 1], face_points_3[:, 2], c='b', marker='o')
    # plot_line(corner_point, intersection_lines[0],ax,color='b')
    # plot_line(corner_point, intersection_lines[1],ax,color='b')
    # plot_line(corner_point, intersection_lines[2],ax,color='b')
    plot_line(corner_point, cube_vectors[:,0],ax)
    plot_line(corner_point, cube_vectors[:,1],ax)
    plot_line(corner_point, cube_vectors[:,2],ax)

    # print(frame_transform)
    rot_mat = frame_transform[0:3, 0:3]
    pose_vec = frame_transform[0:3, 3]
    # r = Rotation.from_matrix(rot_mat)
    # rot_quat = r.as_quat()
    cube_pose = sr.CartesianPose(
        "cube_obj", [pose_vec[0], pose_vec[1], pose_vec[2]], "wobj0")
    # cube_pose.set_orientation(r.as_quat())
    cube_pose.set_orientation(Quaternion(matrix=rot_mat))

    # plt.show() # uncomment to see plot

    return frame_transform
