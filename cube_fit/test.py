from linalg import *
from plot import *
from cube_fit import find_cube_transform


def add_noise(data: np.ndarray, noise_level: float = 0.1):
    return data + noise_level * (np.random.random(data.shape) - 0.5)


def rotate(data: np.ndarray, x: float = 0., y: float = 0., z: float = 0.):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    return data.dot(Rx).dot(Ry).dot(Rz)


ax = generate_axes()
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# quasi-random rotation
rx = np.random.random() * 0.1
ry = np.random.random() * 0.1
rz = np.random.random() * 0.1

# points on the XY plane
xy_points = np.array([
    [1, 1, 2],
    [1, -1, 2],
    [-1, -1, 2],
    [-1, 1, 2]
])

xy_points = rotate(add_noise(xy_points), rx, ry, rz) * 0.2
ax.scatter(xy_points[:, 0], xy_points[:, 1], xy_points[:, 2], c='r', marker='o')
xy_center, xy_plane_normal = plane_of_best_fit(xy_points)
coeff = plane_coefficients(xy_center, xy_plane_normal)
plot_plane_of_best_fit(xy_points, coeff, ax)

# points on the YZ plane
yz_points = np.array([
    [-2, 1, 1],
    [-2, 1, -1],
    [-2, -1, -1],
    [-2, -1, 1]
])
yz_points = rotate(add_noise(yz_points), rx, ry, rz) * 0.2
ax.scatter(yz_points[:, 0], yz_points[:, 1], yz_points[:, 2], c='g', marker='o')
yz_center, yz_plane_normal = plane_of_best_fit(yz_points)
coeff = plane_coefficients(yz_center, yz_plane_normal)
plot_plane_of_best_fit(yz_points, coeff, ax)

# points on the ZX plane
zx_points = np.array([
    [1, 2, 1],
    [-1, 2, 1],
    [-1, 2, -1],
    [1, 2, -1]
])
zx_points = rotate(add_noise(zx_points), rx, ry, rz) * 0.2
ax.scatter(zx_points[:, 0], zx_points[:, 1], zx_points[:, 2], c='b', marker='o')
zx_center, zx_plane_normal = plane_of_best_fit(zx_points)
coeff = plane_coefficients(zx_center, zx_plane_normal)
plot_plane_of_best_fit(zx_points, coeff, ax)

# fit lines of intersections between each plane y, z -> x
point_on_line_x, direction_vector_x = line_of_intersection_2(zx_center, zx_plane_normal, xy_center, xy_plane_normal,
                                                             0.5 * (zx_center + xy_center))
plot_line(point_on_line_x, direction_vector_x, ax)
ax.scatter(point_on_line_x[0], point_on_line_x[1], point_on_line_x[2], c='k', marker='o')

point_on_line_y, direction_vector_y = line_of_intersection_2(xy_center, xy_plane_normal, yz_center, yz_plane_normal,
                                                             0.5 * (xy_center + yz_center))
plot_line(point_on_line_y, direction_vector_y, ax)
ax.scatter(point_on_line_y[0], point_on_line_y[1], point_on_line_y[2], c='k', marker='o')

point_on_line_z, direction_vector_z = line_of_intersection_2(yz_center, yz_plane_normal, zx_center, zx_plane_normal,
                                                             0.5 * (yz_center + zx_center))
plot_line(point_on_line_z, direction_vector_z, ax)
ax.scatter(point_on_line_z[0], point_on_line_z[1], point_on_line_z[2], c='k', marker='o')

# find three pair-wise points of intersections between the lines
p1 = line_line_intersection(point_on_line_x, direction_vector_x, point_on_line_y, direction_vector_y)
p2 = line_line_intersection(point_on_line_y, direction_vector_y, point_on_line_z, direction_vector_z)
p3 = line_line_intersection(point_on_line_z, direction_vector_z, point_on_line_x, direction_vector_x)

corner_point = np.mean(np.array([p1, p2, p3]), axis=0)

corner_point_2 = find_cube_transform(xy_points, yz_points, zx_points)

ax.scatter(corner_point[0], corner_point[1], corner_point[2], c='k', marker='o', s=100)
ax.scatter(corner_point_2[0], corner_point_2[1], corner_point_2[2], c='r', marker='o', s=50)

plt.show()
