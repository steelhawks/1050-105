import numpy as np
import cv2


def resize(input, width, height, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(input, ((int)(width), (int)(height)), 0, 0, interpolation)


def median_filter(input, blur=5):
    return cv2.medianBlur(input,blur)


def hsl_threshold(input,profile):
    hue = profile.hsl_hue
    sat = profile.hsl_sat
    lum = profile.hsl_lum

    out = cv2.cvtColor(input, cv2.COLOR_BGR2HLS)
    return cv2.inRange(out, (hue.min, lum.min, sat.min),  (hue.max, lum.max, sat.max))


def hsv_threshold(input,profile):
    hue = profile.hsv_hue
    sat = profile.hsv_sat
    val = profile.hsv_val

    out = cv2.cvtColor(input, cv2.COLOR_RGB2HSV)
    return cv2.inRange(out, (hue.min, sat.min, val.min),  (hue.max, sat.max, val.max))


def rgb_threshold(input, profile):
    rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb,
                       (profile.red.min, profile.green.min, profile.blue.min),
                       (profile.red.max, profile.green.max, profile.blue.max))
    return mask

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
