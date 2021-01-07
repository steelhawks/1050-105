import cv2
import math
###

###
def dimensions_match(contour, vertices, range, desired_ratio):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    # print(len(approx))
    # if
    if vertices - range <= len(approx) <= vertices + range:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        # print(w)
        # print(h)
        if (desired_ratio < 1 and w > h) or (desired_ratio > 1 and h > w):
            temp = w
            w = h
            h = temp
        MIN_RATIO = desired_ratio * 0.60
        MAX_RATIO = desired_ratio * 1.4
        #MIN_RATIO = desired_ratio * 0.6
        #MAX_RATIO = desired_ratio * 1.4

        ar = w / float(h)
        # print(ar)
        #
        if MIN_RATIO <= ar <= MAX_RATIO:
            return True

    return False


def find_vertices(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    return len(approx)


def get_angle(camera, x, y):

    a = float(abs(camera.FRAME_WIDTH / 2 - x))
    b = float(camera.FRAME_HEIGHT - y)

    if b == 0:
        return 0

    radians = math.atan(a / b)
    angle = radians * 180 / math.pi
    return angle


def get_distance(width_pixel, width_actual, focal_length):
    return focal_length * width_actual / width_pixel

def distance_in_inches(width_pixel):
    return 762 * (width_pixel ** -0.8)

def distance_in_inches_long(width_pixel):
    return 34618 * (width_pixel ** -1.06)
