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


def substitute_in_triangle(im1, tri1, im2, tri2):
    x_min = int(min([vert[0] for vert in tri1]))
    x_max = int(max([vert[0] for vert in tri1]))
    y_min = int(min([vert[1] for vert in tri1]))
    y_max = int(max([vert[1] for vert in tri1]))
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = np.array([x, y])
            s, t = get_coordinates(point, tri1)
            if is_point_in_triangle(s, t):
                im1[x, y] = get_pixel(im2, tri2, s, t)
    return im1

    

file_name = "mountain.jpg"

if __name__ == "__main__":
    im = Image.open(file_name)
    ar = image_to_array(im)
    ar_copy = image_to_array(im)
    
    X, Y, colours = ar.shape
    assert colours == 3

    vert1_1 = np.array([int(0.3*X), int(0.5*Y)])
    vert1_2 = np.array([int(0.5*X), int(0.8*Y)])
    vert1_3 = np.array([int(0.1*X), int(0.9*Y)])
    triangle_1 = [vert1_1, vert1_2, vert1_3]

    vert2_1 = np.array([int(0.9*X), int(0.2*Y)])
    vert2_2 = np.array([int(0.8*X), int(0.3*Y)])
    vert2_3 = np.array([int(0.7*X), int(0.1*Y)])
    triangle_2 = [vert2_1, vert2_2, vert2_3]

    ar = substitute_in_triangle(ar, triangle_1, ar_copy, triangle_2)
    ar = substitute_in_triangle(ar, triangle_2, ar_copy, triangle_1)



    pyplot.imshow(ar)
    pyplot.show()

