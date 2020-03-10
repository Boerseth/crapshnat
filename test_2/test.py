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
    face_landmarks = fr.face_landmarks(fr.load_image_file(file_name))[0]

    ar = image_to_array(im)
    for part, points in face_landmarks.items():
        if "ffeye" in part:
            line_segments = zip(points, points[1:] + [points[0]])
        else:
            line_segments = zip(points[:-1], points[1:])
        for p1, p2 in line_segments:
            x1, y1 = p1
            x2, y2 = p2
            for i in range(3):
                if abs(x2 - x1) > abs(y2 - y1):
                    k1, k2, l1, l2 = (x1, x2, y1, y2) if (x2 > x1) else (x2, x1, y2, y1)
                    for k in range(k1, k2+1):
                        theta = (k - k1) / (k2 - k1)
                        l = int((1 - theta) * l1 + theta * l2)
                        for j in range(-1, 2):
                            ar[l+j, k, i] = 255
                else:
                    k1, k2, l1, l2 = (x1, x2, y1, y2) if (y2 > y1) else (x2, x1, y2, y1)
                    for l in range(l1, l2+1):
                        theta = (l - l1) / (l2 - l1)
                        k = int((1 - theta) * k1 + theta * k2)
                        for j in range(-1, 2):
                            ar[l, k+j, i] = 255
    pyplot.imshow(ar)
    pyplot.show()

