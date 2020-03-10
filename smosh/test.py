import sys

import face_recognition as fr
import numpy as np
from PIL import Image
from matplotlib import pyplot


def image_to_array(im):
    return np.array(im)


def array_to_image(array):
    return Image.fromarray(array)


def get_coordinates(point, triangle):
    p0, p1, p2 = triangle
    p = point - p0
    v1 = p1 - p0
    v2 = p2 - p0
    det = v1[0]*v2[1] - v2[0]*v1[1]
    s = (v2[1]*p[0] - v2[0]*p[1]) / det
    t = (v1[0]*p[1] - v1[1]*p[0]) / det
    return s, t


def is_point_in_triangle(s, t):
    return (0 <= s <= 1) and (0 <= t <= 1) and (s+t <= 1)


def weight(a, b, w):
    return a*(1-w) + b*w


def get_pixel(im, tri, s, t):
    p0, p1, p2 = tri
    point = p0 + s*(p1-p0) + t*(p2-p0)
    x_, y_ = point[0], point[1]
    x, dx = int(x_), x_%1
    y, dy = int(y_), y_%1
    return weight(
        weight(im[x,y], im[x+1,y], dx),
        weight(im[x,y+1], im[x+1,y+1], dx),
        dy
    )


def get_boundaries(tri):
    x_min = int(min([vert[0] for vert in tri]))
    x_max = int(max([vert[0] for vert in tri]))
    y_min = int(min([vert[1] for vert in tri]))
    y_max = int(max([vert[1] for vert in tri]))
    return x_min, x_max, y_min, y_max


alpha = 5.0
def g(z):
    return np.arctan(alpha*z)/np.pi + 0.5


def smosh(s, t):
    return (
        1.5*s*(1-g((s+t)-0.7)) + s*g((s+t)-0.7), 
        1.5*t*(1-g((s+t)-0.7)) + t*g((s+t)-0.7)
    )


def smosh_triangle(im1, tri1, im2, tri2):
    x_min, x_max, y_min, y_max = get_boundaries(tri1)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = np.array([x, y])
            s, t = get_coordinates(point, tri1)
            if is_point_in_triangle(s, t):
                s, t = smosh(s, t)
                im1[x, y] = get_pixel(im2, tri2, s, t)
    return im1


def get_key_points_from_landmarks(landmarks):
    chin = landmarks["chin"]
    nose_bridge = landmarks["nose_bridge"]
    left_eyebrow = landmarks["left_eyebrow"]
    right_eyebrow = landmarks["right_eyebrow"]
    center = nose_bridge[3]
    key_points = chin + right_eyebrow[::-1] + left_eyebrow[::-1]
    # key_points = (
    #         list(chin[i] for i in [0, 4, 6, 8, 10, 12, 16])
    #         + list(right_eyebrow[i] for i in [4, 2, 0])
    #         + list(left_eyebrow[i] for i in [4, 2, 0])
    # )
    return np.array([center[1], center[0]]), [np.array([kp[1], kp[0]]) for kp in key_points]


def get_triangles(center, orbits):
    return [
        [center, orbit1, orbit2] for orbit1, orbit2 in zip(orbits, orbits[1:] + [orbits[0]])
    ]


if __name__ == "__main__":
    file_name = sys.argv[1]

    face_lands = fr.face_landmarks(fr.load_image_file(file_name))
    face_landmarks = face_lands[0]
    center, key_points = get_key_points_from_landmarks(face_landmarks)
    triangles = get_triangles(center, key_points)

    im = Image.open(file_name)
    ar = image_to_array(im)
    ar_copy = image_to_array(im)

    for triangle in triangles:
        ar = smosh_triangle(ar, triangle, ar_copy, triangle)

    pyplot.imshow(ar)
    pyplot.show()

