from cube_fit.plot import *


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

# points on the YZ plane
yz_points = np.array([
    [-2, 1, 1],
    [-2, 1, -1],
    [-2, -1, -1],
    [-2, -1, 1]
])
yz_points = rotate(add_noise(yz_points), rx, ry, rz) * 0.2

# points on the ZX plane
zx_points = np.array([
    [1, 2, 1],
    [-1, 2, 1],
    [-1, 2, -1],
    [1, 2, -1]
])
zx_points = rotate(add_noise(zx_points), rx, ry, rz) * 0.2

# plot all points, planes and lines
plot_cube_fit(xy_points, yz_points, zx_points)
