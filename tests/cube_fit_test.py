from cube_fit.cube_fit import *
import numpy as np


# data of cube on table to compare similarity of results
face_points_1 = np.array([[-0.03945581769,	-0.03158745261,	0.0001617736816],
                          [-0.02586010583,	-0.04625624184,	-0.0005893974304],
                          [-0.05528479484,	-0.04646567142,	0.0003114585876],
                          [-0.03963978829,	-0.06141738854,	0.0001161651611]])
face_points_2 = np.array([[-0.01941319041,	0.02749663343, -0.01018671036],
                          [0.01531810512, 0.02568523063,	-0.01010115814],
                          [-0.05430244026,	0.0292663148,	-0.01033506012],
                          [-0.01958680416,	0.02755419873, -0.0249793396]])
face_points_3 = np.array([[0.02496873613, -0.01365341988, -0.01078731155],
                          [0.0264051176, 0.01122091341,	-0.01072323608],
                          [0.02479365047, -0.01389522632, -0.02556806946],
                          [0.02359494505, -0.03871262625, -0.01087914658]])
a = find_cube_transform(face_points_1, face_points_2, face_points_3)
b = six_point_cube_transform(face_points_1[0:3], face_points_2[0:2], face_points_3[0:1])
se3_difference(a,b)

# manufactured data for correctness
face_points_1 = np.array([
    [0,3,4],
    [0,2,1],
    [0,1,2]
])
face_points_2 = np.array([
    [1,2,0],
    [2,1,0],
    [4,3,0]
])
face_points_3 = np.array([
    [1,0,2],
    [5,0,2],
    [2,0,1]
])
a = find_cube_transform(face_points_1, face_points_2, face_points_3)
b = six_point_cube_transform(face_points_1[0:3], face_points_2[0:2], face_points_3[0:1])
se3_difference(a,b)

# data of cube on MiR 
face_points_1 = np.array([[-0.04024476914, -0.03293735979, -3.87E-05],
                          [-0.02512830387, -0.04766131331, -6.74E-05],
                          [-0.05538585215, -0.04778255944, -5.50E-05],
                          [-0.04040184808, -0.0628618676, 0.0001611814499]])

face_points_2 = np.array([[-0.0184927499, 0.02663407422, -0.01167119694],
                          [0.01654984701, 0.0250818916, -0.01147095776],
                          [-0.05355432354, 0.02807404167, -0.01178652477],
                          [-0.01858676236, 0.02659035496, -0.02665022135]])

face_points_3 = np.array([[0.02883716861, -0.01707960727, -0.007740744591],
                          [0.0298599217, 0.007818343067, -0.008041884422],
                          [0.0290603834, -0.0170210296, -0.02296379757],
                          [0.02771588624, -0.04182889291, -0.007894488335]])

a = find_cube_transform(face_points_1, face_points_2, face_points_3)
b = six_point_cube_transform(face_points_1[0:3], face_points_2[0:2], face_points_3[0:1])
se3_difference(a,b)

