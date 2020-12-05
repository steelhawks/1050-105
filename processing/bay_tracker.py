"""
2020 Frc Infinite Recharge
Bay Intake Detection
uses contour lines and 
width/length ratios, area, 
and vertices to check a masked image
for the bay
same as port tracker but with diff.
object dimensions
"""

import math
import cv2
from processing import colors
from processing import cvfilters
from processing import shape_util
import time

from profiles import color_profiles 

MIN_AREA = 50
BAY_LENGTH = 7

# 
WIDTH_TO_HEIGHT_RATIO = 7 / 11 

debug = False

def process(img, camera, frame_cnt, color_profile):
    global rgb_window_active, hsv_window_active
    FRAME_WIDTH = camera.FRAME_WIDTH
    FRAME_HEIGHT = camera.FRAME_HEIGHT
    hue = color_profile.hsv_hue
    sat = color_profile.hsv_sat
    val = color_profile.hsv_val

    tracking_data = []
    original_img = img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (hue.min, sat.min, val.min),  (hue.max, sat.max, val.max))
    img = cvfilters.apply_mask(img, mask)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if debug:
    #     cv2.imshow('hsv', img)

    contours, hierarchy = cv2.findContours(img,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    # algorithm for detecting rectangular object (loading bay)
    for (index, contour) in enumerate(contours):

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        area = cv2.contourArea(approx)
        x, y, w, h = cv2.boundingRect(approx)
        # limit the number of contours to process
        #
        num_vertices = shape_util.find_vertices(contour)
        if area > MIN_AREA:
            contour_list.append(contour)
            center_mass_x = x + w / 2
            center_mass_y = y + h / 2
            #
            if shape_util.dimensions_match(contour, 4, 2, WIDTH_TO_HEIGHT_RATIO):
                # print 'x:%s, y:%s angle:%s ' % ( center_mass_x, center_mass_y, angle )
                distance = shape_util.distance_in_inches(w)
                angle = shape_util.get_angle(camera, center_mass_x, center_mass_y)
                font = cv2.FONT_HERSHEY_DUPLEX

                # set tracking_data
                data = dict(shape='BAY',
                        width=w,
                        height=h,
                        dist=distance,
                        angle=angle,
                        xpos=center_mass_x,
                        ypos=center_mass_y)
                
                tracking_data.append(data)
                
                vertices_text = 'vertices:%s' % (num_vertices)
                coordinate_text = 'x:%s y:%s ' % (center_mass_x, center_mass_y)
                area_text = 'area:%s width:%s height:%s' % (area, w, h)
                angle_text = 'angle:%.2f  distance:%.2f' % (angle, distance)

                cv2.putText(original_img, coordinate_text, (x, y - 35), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                cv2.putText(original_img, area_text, (x, y - 20), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                cv2.putText(original_img, angle_text, (x, y - 5), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                cv2.putText(original_img, vertices_text, (x, y - 50), font, .4, colors.WHITE, 1, cv2.LINE_AA)

                cv2.rectangle(original_img, (x, y), (x + w, y + h), colors.GREEN, 2)
                cv2.drawContours(original_img, contours, index, colors.random(), 2)
                #cv2.circle(original_img, (int(center_mass_x), int(center_mass_y)), 5, colors.GREEN, -1)
                cv2.line(original_img, (FRAME_WIDTH // 2, FRAME_HEIGHT), (int(center_mass_x), int(center_mass_y)), colors.GREEN, 2)
            # elif debug:
                
                # cv2.drawContours(original_img, contours, index, colors.random(), 2)
                #cv2.rectangle(original_img, (x, y), (x + w, y + h), colors.WHITE, 2)

                # print the rectangle that did not match

            #
            # print 'square: %s,%s' % (w,h)
            # print w/h, h/w
    top_center = (FRAME_WIDTH // 2, FRAME_HEIGHT)
    bottom_center = (FRAME_WIDTH // 2, 0)
    cv2.line(original_img, top_center, bottom_center, colors.WHITE, 4)
    return original_img, tracking_data