"""
2019 Frc Deep Space
Bay Detection
uses contour lines and 
width/length ratios, area, 
and vertices to check a masked image
for the bay
"""

import cv2
import numpy as np
from processing import colors
from processing import cvfilters
from processing import shape_util

import controls
from controls import main_controller
import uuid

MIN_AREA = 100
BAY_LENGTH = 7

LEFT_STRIP = 'LEFT'
RIGHT_STRIP = 'RIGHT'

WIDTH_TO_HEIGHT_RATIO = 3.3 / 5.8 

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

    _, contours, hierarchy = cv2.findContours(img,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    contour_properties_list = []
    # algorithm for detecting rectangular object (loading bay)
    for (index, contour) in enumerate(contours):

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        area = cv2.contourArea(approx)
        x, y, w, h = cv2.boundingRect(approx)
        # _, (MA, ma), orientation = cv2.fitEllipse(contour)
        # print(orientation)
        rect = cv2.minAreaRect(contour)
        orientation = rect[2]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(original_img, [box], 0, (0, 0, 255), 2)
        cv2.putText(original_img, str(orientation), (x, y + h + 35), cv2.FONT_HERSHEY_DUPLEX, .4, colors.WHITE, 1, cv2.LINE_AA)
        #print(orientation)
        # limit the number of contours to process
        
        num_vertices = shape_util.find_vertices(contour)
        if area > MIN_AREA:
            orientation = abs(orientation)
            strip_type = 'NOT VALID'
            if orientation > 2.5 and orientation < 25:
                strip_type = RIGHT_STRIP
                # print(RIGHT_STRIP)
            elif orientation > 65 and orientation < 87.5:
                strip_type = LEFT_STRIP
                # print(LEFT_STRIP)

            contour_list.append(contour)
            contour_properties_list.append([x, y, w, h, strip_type])

            if strip_type != 'NOT_VALID':

                other_contour_properties = None

                i = len(contour_properties_list) - 1

                while i >= 0:
                    if strip_type == LEFT_STRIP and contour_properties_list[i][4] != LEFT_STRIP:
                        if contour_properties_list[i][0] > x:
                            # print('found strip')
                            # print(strip_type)
                            # print(contour_properties_list[i])
                            # print('----')
                            other_contour_properties = contour_properties_list[i]
                            contour_properties_list.pop(i)
                            i = 0
                    elif strip_type == RIGHT_STRIP and contour_properties_list[i][4] != RIGHT_STRIP:
                        if contour_properties_list[i][0] < x:
                            # print('found strip')
                            # print(strip_type)
                            # print(contour_properties_list[i])
                            # print('----')
                            other_contour_properties = contour_properties_list[i]
                            contour_properties_list.pop(i)
                            i = 0    
                    i = i - 1

                if other_contour_properties != None:
                    
                    center_mass_x_other = other_contour_properties[0] + other_contour_properties[2] / 2
                    center_mass_y_other = other_contour_properties[1] + other_contour_properties[3] / 2

                    center_mass_x_right = x + w / 2
                    center_mass_y_right = y + h / 2

                    center_mass_x_avg = (center_mass_x_other + center_mass_x_right) / 2
                    center_mass_y_avg = (center_mass_y_other + center_mass_y_right) / 2

                    width = 0
                    xstart = 0
                    xend = 0
                    height = 0

                    if strip_type == RIGHT_STRIP:

                        width = x - other_contour_properties[0] + w
                        xstart = other_contour_properties[0]
                        xend = other_contour_properties[0] + width
                        height = h

                    elif strip_type == LEFT_STRIP:

                        width = other_contour_properties[0] - x + other_contour_properties[2]
                        xstart = x
                        xend = x + width
                        height = h

                    distance = shape_util.distance_in_inches((abs(x - other_contour_properties[0]) + w))
                    angle = shape_util.get_angle(camera, center_mass_x_avg, center_mass_y_avg)
                    
                    if isinstance(distance, complex):
                        distance = 1

                    font = cv2.FONT_HERSHEY_DUPLEX

                    data = dict(shape='BAY',
                            width=width,
                            height=height,
                            dist=distance,
                            angle=angle,
                            xpos=center_mass_x_avg,
                            ypos=center_mass_y_avg)
                    
                    tracking_data.append(data)

                    vertices_text = 'vertices:%s' % (num_vertices)
                    coordinate_text = 'x:%s y:%s ' % (center_mass_x_avg, center_mass_y_avg)
                    area_text = 'width:%s height:%s' % (width, height)
                    angle_text = 'angle:%.2f  distance:%.2f' % (angle, distance)

                    cv2.putText(original_img, coordinate_text, (x, y - 35), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, area_text, (x, y - 20), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, angle_text, (x, y - 5), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, vertices_text, (x, y - 50), font, .4, colors.WHITE, 1, cv2.LINE_AA)

                    cv2.rectangle(original_img, (xstart, y), (xend, height + y), colors.GREEN, 2)
                    #cv2.drawContours(original_img, contours, index, colors.random(), 2)
                    #cv2.circle(original_img, (int(center_mass_x), int(center_mass_y)), 5, colors.GREEN, -1)
                    cv2.line(original_img, (FRAME_WIDTH // 2, FRAME_HEIGHT), (int(center_mass_x_avg), int(center_mass_y_avg)), colors.GREEN, 2)
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

