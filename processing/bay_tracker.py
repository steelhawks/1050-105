"""
2019 Frc Deep Space
Bay Detection
uses contour lines and 
width/length ratios, area, 
and vertices to check a masked image
for the bay
"""

import cv2
from processing import colors
from processing import cvfilters
from processing import shape_util

MIN_AREA = 50
BAY_LENGTH = 7

LEFT_STRIP = 'LEFT'
RIGHT_STRIP = 'RIGHT'

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
        print(orientation)
        # limit the number of contours to process
        
        num_vertices = shape_util.find_vertices(contour)
        if area > MIN_AREA:
            strip_type = 'NOT VALID'
            if orientation < -5 and orientation > -25:
                strip_type = RIGHT_STRIP
                print(RIGHT_STRIP)
            elif orientation < -65 and orientation > -85:
                strip_type = LEFT_STRIP
                print(LEFT_STRIP)

            contour_list.append(contour)
            contour_properties_list.append([x, y, w, h, strip_type])
            # shape_util.dimensions_match(contour, 4, 2, WIDTH_TO_HEIGHT_RATIO) and 
            if strip_type == RIGHT_STRIP:

                left_contour_properties = None

                i = len(contour_properties_list) - 1

                while i >= 0:
                    if contour_properties_list[i][4] == LEFT_STRIP:
                        print('found strip')
                        left_contour_properties = contour_properties_list[i]
                        i = 0
                    i = i - 1

                if left_contour_properties != None:
                    
                    center_mass_x_left = left_contour_properties[0] + left_contour_properties[2] / 2
                    center_mass_y_left = left_contour_properties[1] + left_contour_properties[3] / 2

                    center_mass_x_right = x + w / 2
                    center_mass_y_right = y + h / 2

                    center_mass_x_avg = (center_mass_x_left + center_mass_x_right) / 2
                    center_mass_y_avg = (center_mass_y_left + center_mass_y_right) / 2

                    distance = shape_util.distance_in_inches(((x - left_contour_properties[0]) + w))
                    angle = shape_util.get_angle(camera, center_mass_x_avg, center_mass_y_avg)
                    
                    if isinstance(distance, complex):
                        distance = 1

                    font = cv2.FONT_HERSHEY_DUPLEX

                    data = dict(shape='BAY',
                            width=((x - left_contour_properties[0]) + w),
                            height=h,
                            dist=distance,
                            angle=angle,
                            xpos=center_mass_x_avg,
                            ypos=center_mass_y_avg)
                    
                    tracking_data.append(data)

                    vertices_text = 'vertices:%s' % (num_vertices)
                    coordinate_text = 'x:%s y:%s ' % (center_mass_x_avg, center_mass_y_avg)
                    area_text = 'width:%s height:%s' % (((x - left_contour_properties[0]) + w), h)
                    angle_text = 'angle:%.2f  distance:%.2f' % (angle, distance)

                    cv2.putText(original_img, coordinate_text, (x, y - 35), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, area_text, (x, y - 20), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, angle_text, (x, y - 5), font, .4, colors.WHITE, 1, cv2.LINE_AA)
                    cv2.putText(original_img, vertices_text, (x, y - 50), font, .4, colors.WHITE, 1, cv2.LINE_AA)

                    cv2.rectangle(original_img, (left_contour_properties[0], left_contour_properties[1]), (((x - left_contour_properties[0]) + w + left_contour_properties[0]), ((y - left_contour_properties[1]) + h + left_contour_properties[1])), colors.GREEN, 2)
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

