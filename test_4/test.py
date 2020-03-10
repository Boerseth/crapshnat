import face_recognition as fr
import numpy as np
from PIL import Image
from matplotlib import pyplot


def image_to_array(im):
    return np.array(im)


def array_to_image(array):
    return Image.fromarray(array)


def weight(a, b, w):
    return a*(1-w) + b*w


def int_and_trail(r):
    return int(r), r%1


def get_pixel(im, x1, y2, x2, y1, xr, yr):
    x, dx = int_and_trail(x1 + (x2-x1) * xr)
    y, dy = int_and_trail(y1 + (y2-y1) * yr)
    return weight(
        weight(im[x,y], im[x+1,y], dx),
        weight(im[x,y+1], im[x+1,y+1], dx),
        dy,
    )


def overwrite_face(ar1, ar2, face_loc_1, face_loc_2):
    print(face_loc_1)
    x1, y2, x2, y1 = face_loc_1
    for x in range(x1, x2):
        for y in range(y1, y2):
            xr = (x-x1)/(x2-x1)
            yr = (y-y1)/(y2-y1)
            ar1[x, y] = get_pixel(ar2, *face_loc_2, xr, yr)
    return ar1


def swap_faces(ar, ar_copy, face_locations):
    face_loc_1, face_loc_2 = face_locations[0], face_locations[1]
    ar = overwrite_face(ar, ar_copy, face_loc_1, face_loc_2)
    ar = overwrite_face(ar, ar_copy, face_loc_2, face_loc_1)
    return ar

file_name = "obama_trump2.jpg"

if __name__ == "__main__":
    im = Image.open(file_name)
    # y1, x2, y2, x1 
    # print(f"{x1}, {x2}, {y1}, {y2}")
    face_locations = fr.face_locations(fr.load_image_file(file_name))

    ar = image_to_array(im)
    ar_copy = image_to_array(im)
    ar = swap_faces(ar, ar_copy, face_locations)

    pyplot.imshow(ar)
    pyplot.show()

