import face_recognition as fr
import numpy as np
from PIL import Image
from matplotlib import pyplot


def image_to_array(im):
    return np.array(im)


def array_to_image(array):
    return Image.fromarray(array)


file_name = "obama.jpg"

if __name__ == "__main__":
    im = Image.open(file_name)
    y1, x2, y2, x1 = fr.face_locations(fr.load_image_file(file_name))[0]
    print(f"{x1}, {x2}, {y1}, {y2}")
    assert y2 > y1
    assert x2 > x1

    ar = image_to_array(im)
    for i in range(3):
        for j in range(5):
            for x in range(x1, x2):
                ar[y1+j, x, i] = 255
                ar[y2+j, x, i] = 255
            for y in range(y1, y2):
                ar[y, x1+j, i] = 255
                ar[y, x2+j, i] = 255
    pyplot.imshow(ar)
    pyplot.show()

