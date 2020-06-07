import numpy as np
import cv2


def resize(input, width, height, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(input, ((int)(width), (int)(height)), 0, 0, interpolation)


def median_filter(input, blur=5):
    return cv2.medianBlur(input,blur)


def hsl_threshold(input,profile):
    hue = profile.hue
    sat = profile.sat
    lum = profile.lum

    out = cv2.cvtColor(input, cv2.COLOR_BGR2HLS)
    return cv2.inRange(out, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))


def hsv_threshold(input,profile):
    hue = profile.hsv_hue
    sat = profile.hsv_sat
    val = profile.hsv_val

    out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))


def rgb_threshold(input, profile):


    red = profile.red
    green = profile.green
    blue = profile.blue

    out = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    return cv2.inRange(out, (red[0], green[0], blue[0]),  (red[1], green[1], blue[1]))


def grayscale(input):
    """
    convert to grayscale
    """
    return cv2.cvtColor(input,cv2.COLOR_RGB2GRAY)


def noise_removal(input):
    """
    Noise removal with iterative
    bilateral filter(removes noise while preserving edges)
    """
    return cv2.bilateralFilter(input, 9,75,75)


def detect_canny_edges(input, debug=False):
    """
    Applying Canny Edge detection
    """

    canny_image = cv2.Canny(input,250,255)
    if debug:
        cv2.imshow("Image after applying Canny",canny_image)
    # # Display Image
    return cv2.convertScaleAbs(canny_image)


def dilate_edges(img):
    # dilation to strengthen the edges
    kernel = np.ones((3,3), np.uint8)
    # Creating the kernel for dilation
    dilated_image = cv2.dilate(img,kernel,iterations=1)
    return dilated_image


def threshold_OTSU(img):
    # Display Image
    # Thresholding the image
    _,thresh_image = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

    return thresh_image



def apply_mask(input, mask):
    """Filter out an area of an image using a binary mask.
    Args:
        input: A three channel numpy.ndarray.
        mask: A black and white numpy.ndarray.
    Returns:
        A three channel numpy.ndarray.
    """
    return cv2.bitwise_and(input, input, mask=mask)