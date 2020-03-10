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


def hsl_from_rgb(r, g, b):
    R, G, B = r/255, g/255, b/255
    Cmax = max(R, G, B)
    Cmin = min(R, G, B)
    D = Cmax - Cmin
    L = (Cmax + Cmin)/2
    H = 0
    if D == 0:
        H = 0
        S = 0
    elif Cmax == R:
        H = 60 * (((G - B)/D) % 6)
    elif Cmax == G:
        H = 60 * (((B - R)/D) + 2)
    elif Cmax == B:
        H = 60 * (((R - G)/D) + 4)
    if H < 0:
        H += 360
    S = D / (1 - abs(2*L - 1))
    return H, S, L


def rgb_from_hsl(h, s, l):
    C = (1 - abs(2*l - 1)) * s
    X = C * (1 - abs((h/60) % 2 - 1))
    m = l - C/2
    R = [C, X, 0, 0, X, C][int(h / 60)]
    G = [X, C, C, X, 0, 0][int(h / 60)]
    B = [0, 0, X, C, C, X][int(h / 60)]
    return int((R + m)*255), int((G + m)*255), int((B + m)*255)


def gradual(z):
    if z < 0.8:
        return 1
    else:
        return 1 - 5 * (z - 0.8)

def combine_pixels(r, g, b, r0, b0, g0, rho_l, alpha):
    h, s, l = hsl_from_rgb(r, g, b)
    h0, s0, l0 = hsl_from_rgb(r0, g0, b0)
    return np.array([r0, g0, b0])*(1-alpha) + np.array(rgb_from_hsl(h0, s, min(0.99, rho_l*l)))*alpha


def get_pixel(im, tri, s, t, pixel0, rho_l):
    p0, p1, p2 = tri
    point = p0 + s*(p1-p0) + t*(p2-p0)
    x_, y_ = point[0], point[1]
    x, dx = int(x_), x_%1
    y, dy = int(y_), y_%1
    pixel = weight(
        weight(im[x,y], im[x+1,y], dx),
        weight(im[x,y+1], im[x+1,y+1], dx),
        dy
    )
    r, g, b = pixel[0], pixel[1], pixel[2]
    r0, g0, b0= pixel0[0], pixel0[1], pixel0[2]
    alpha = gradual(s + t)
    return combine_pixels(r, g, b, r0, b0, g0, rho_l, alpha)


def get_boundaries(tri):
    x_min = int(min([vert[0] for vert in tri]))
    x_max = int(max([vert[0] for vert in tri]))
    y_min = int(min([vert[1] for vert in tri]))
    y_max = int(max([vert[1] for vert in tri]))
    return x_min, x_max, y_min, y_max


def substitute_in_triangle(im1, tri1, im2, tri2, rho_l):
    x_min, x_max, y_min, y_max = get_boundaries(tri1)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = np.array([x, y])
            s, t = get_coordinates(point, tri1)
            if is_point_in_triangle(s, t):
                im1[x, y] = get_pixel(im2, tri2, s, t, im1[x, y], rho_l)
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


def get_avg_luminosity(im, triangles):
    l_vals = []
    for tri in triangles:
        x_min, x_max, y_min, y_max = get_boundaries(tri)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                point = np.array([x, y])
                s, t = get_coordinates(point, tri)
                if is_point_in_triangle(s, t):
                    h, s, l = hsl_from_rgb(*list(im[x,y]))
                    l_vals.append(l)
    assert l_vals
    return sum(l_vals) / len(l_vals)
    

if __name__ == "__main__":
    file_name = sys.argv[1]

    face_lands = fr.face_landmarks(fr.load_image_file(file_name))
    face_landmarks_1 = face_lands[0]
    face_landmarks_2 = face_lands[1]
    center_1, key_points_1 = get_key_points_from_landmarks(face_landmarks_1)
    center_2, key_points_2 = get_key_points_from_landmarks(face_landmarks_2)
    triangles_1 = get_triangles(center_1, key_points_1)
    triangles_2 = get_triangles(center_2, key_points_2)

    im = Image.open(file_name)
    ar = image_to_array(im)
    ar_copy = image_to_array(im)

    L1 = get_avg_luminosity(ar, triangles_1)
    L2 = get_avg_luminosity(ar, triangles_2)
    rho_l1 = np.sqrt(L1/L2)
    rho_l2 = np.sqrt(L2/L1)

    for triangle_1, triangle_2 in zip(triangles_1, triangles_2):
        ar = substitute_in_triangle(ar, triangle_1, ar_copy, triangle_2, rho_l1)
        ar = substitute_in_triangle(ar, triangle_2, ar_copy, triangle_1, rho_l2)

    pyplot.imshow(ar)
    pyplot.show()

