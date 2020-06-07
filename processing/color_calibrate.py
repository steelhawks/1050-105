"""

Object Tracker based on a color profile

uses contour lines and rough area calculations

"""

import cv2
from controls import main_controller
from . import colors
from processing import cvfilters

def process(image,
            camera_mode='RAW',
            color_mode='rgb',
            apply_mask=False):

    image = cv2.resize(image, ((int)(640), (int)(480)), 0, 0, cv2.INTER_CUBIC)


    if camera_mode != 'RAW':

        color_profile = main_controller.color_profiles.get(camera_mode)

        mask = None

        if color_mode == 'rgb':

            mask = cv2.inRange(image,
                               (color_profile.red.min, color_profile.green.min, color_profile.blue.min),
                               (color_profile.red.max, color_profile.green.max, color_profile.blue.max))

        elif color_mode == 'hsv':
            hue = color_profile.hsv_hue
            sat = color_profile.hsv_sat
            val = color_profile.hsv_val

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (hue.min, sat.min, val.min),  (hue.max, sat.max, val.max))

        if mask is not None:

            if apply_mask:
                image = cvfilters.apply_mask(image, mask)
                image = cv2.erode(image, None, iterations=2)
                image = cv2.dilate(image, None, iterations=2)
            else:
                image = mask

    cv2.putText(image,
            'Mode %s' % camera_mode,
            (20,20),
            cv2.FONT_HERSHEY_DUPLEX,
            .4,
            colors.BLUE,
            1,
            cv2.LINE_AA)

    if color_mode is not None:
        cv2.putText(image,
                    'COLOR Mode %s' % color_mode,
                    (20,40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    .4,
                    colors.BLUE,
                    1,
                    cv2.LINE_AA)

    return image
