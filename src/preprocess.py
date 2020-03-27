'''
This script includes methods of manually aligning the stack of images.
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab, uOttawa
'''
import os
import cv2
import numpy as np
import copy
import enum
import math
# Global variable for opencv painting method.
drawing = False
x1, y1, x2, y2 = 0, 0, 0, 0

class status(enum.Enum):
    """
    Identifying the current status for human labeling.
    """
    point1 = 0
    point2 = 1
    vertical_line = 3
    critical_line = 4

class Affine:
    """
    data structure for storing necessary data for affining.
    """
    def __init__(self):
        self.ponit1 = Point()
        self.ponit2 = Point()
    def isInstance(self):
        return self.ponit1.isInstance and self.ponit2.isInstance

class Line:
    """
    data structure for storing line data.
    """
    def __init__(self, x1=None, y1=None, x2=None, y2=None, pos=None):
        self.point1 = Point(x1, y1)
        self.point2 = Point(x2, y2)
        self.pos = pos
    def isInstance(self):
        # When the two points in a line are both a real instance, return True.
        return self.point1.isInstance and self.point2.isInstance

class Point:
    """
    basic data structure for storing point.
    """
    def __init__(self, x=None, y=None):
        self._x = x
        self._y = y
        # isInstance property will and only will be set as True when x and y are both set as valid values.
        self.isInstance = False

    @property
    def x(self):
        return self._x

    # Trigger setter method when assigning value.
    @x.setter
    def x(self, value):
        self._x = value
        if self._x and self._y:
            self.isInstance = True
        else:
            self.isInstance = False

    @property
    def y(self):
        return self._y

    # Trigger setter method when assigning value.
    @y.setter
    def y(self, value):
        self._y = value
        if self._y and self._y:
            self.isInstance = True
        else:
            self.isInstance = False

def click_and_draw_line(event, x, y, flags, param):
    """
    Method for painting in opencv, will be used in cv2.setMouseCallback("Window", click_and_draw_line)
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global x1, y1, x2, y2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        x2 = x
        y2 = y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if(drawing):
            x2 = x
            y2 = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def prepare_tissue_image(input_folder, output_folder, color_channel="b", section_thickness=100):
    '''
    This is the first step to get image aligned and also get the origin point.
    :param input_folder: The folder contains a brain.
    :param output_folder: The output folder for this brain.
    :return:
    '''
    color_channel_dict = {'b': 0, 'g': 1, 'r': 2}

    img_paths = [os.path.join(input_folder, item) for item in os.listdir(input_folder)
                 if item.endswith('_%s.jpg' % color_channel)]

    def sort_key(item):
        item = os.path.basename(item)
        hour = int(item.split('.')[0].split(':')[0].split('_')[1])
        min = int(item.split('.')[0].split(':')[1])
        sec = int(item.split('.')[0].split(':')[2].split('_')[0])
        return ((hour * 100) + min) * 100 + sec
    img_paths.sort(key=sort_key)

    # raw_images = [cv2.imread(item) for item in img_paths]
        top, bottom, left, right = 400, 100, 100, 100
    raw_images = [cv2.copyMakeBorder(cv2.imread(item), top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0)) for item in img_paths] 
    stack_length = len(raw_images)
    pos = 0
    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", click_and_draw_line)
    status_dict = {}
    affine_dict = {}
    critical_line = Line()
    vertical_line = Line()
    manual_dict = {status.point1: "the point matches the previous image",
                   status.point2: "the point matches the next image",
                   status.vertical_line: "a vertical line from top to bottom",
                   status.critical_line: "the critical line in brain"}
    color_dict = {
                    status.point1: (255, 0, 0),
                    status.point2: (0, 255, 0),
                    status.vertical_line: (125, 125, 0),
                    status.critical_line: (0, 125, 125)
                  }
    while(True):
        if pos not in affine_dict.keys():
            affine_dict[pos] = Affine()
            if pos == 0:
                affine_dict[pos].ponit1.x = 1
                affine_dict[pos].ponit1.y = 1
        if pos not in status_dict.keys():
            status_dict[pos] = status.point1
            if pos == 0:
                status_dict[pos] = status.point2


        show_image = copy.deepcopy(raw_images[pos])
        show_image = cv2.resize(show_image, (int(show_image.shape[1] * 0.5), int(show_image.shape[0] * 0.5)))

        while (True):
            if vertical_line.isInstance() and affine_dict[pos].isInstance():
                status_dict[pos] = status.critical_line

            finshed_flag = False
            canvas = copy.deepcopy(show_image)
            if status_dict[pos].value == 1:
                cv2.line(canvas, (x1, y1), (x1, y1), (0, 0, 255), thickness=3)
            elif status_dict[pos].value == 2:
                cv2.line(canvas, (x1, y1), (x1, y1), (0, 0, 255), thickness=3)
            else:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

            if affine_dict[pos].ponit1.isInstance:
                cv2.line(canvas, (affine_dict[pos].ponit1.x, affine_dict[pos].ponit1.y),
                         (affine_dict[pos].ponit1.x, affine_dict[pos].ponit1.y), (0, 255, 0), thickness=5)

            if affine_dict[pos].ponit2.isInstance:
                cv2.line(canvas, (affine_dict[pos].ponit2.x, affine_dict[pos].ponit2.y),
                         (affine_dict[pos].ponit2.x, affine_dict[pos].ponit2.y), (0, 255, 0), thickness=5)

            cv2.putText(canvas, "Pos: %3d" % pos, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            for key in color_dict.keys():
                color_dict[key] = (85, 85, 85)
            color_dict[status_dict[pos]] = (0, 255, 0)

            cv2.putText(canvas, "1. Draw %s, press p to save" % manual_dict[status.point1], (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.point1], 2, cv2.LINE_AA)
            cv2.putText(canvas, "2. Draw %s, press p to save" % manual_dict[status.point2], (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.point2], 2, cv2.LINE_AA)
            cv2.putText(canvas, "3. Draw %s, press p to save" % manual_dict[status.vertical_line], (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.vertical_line], 2, cv2.LINE_AA)
            cv2.putText(canvas, "4. Draw %s, press p to save" % manual_dict[status.critical_line], (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.critical_line], 2, cv2.LINE_AA)
            cv2.putText(canvas, "5. Press Q to save all and start to calibrate", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.critical_line], 2, cv2.LINE_AA)
            cv2.putText(canvas, "Others: Press A and D to go previous or next", (50, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[status.critical_line], 2, cv2.LINE_AA)
            cv2.imshow("Window", canvas)

            key = cv2.waitKey(20)
            if key & 0xFF == ord('a'):
                pos -= 1
                if pos == -1:
                    pos = len(raw_images) - 1
                break
            elif key & 0xFF == ord('d'):
                pos += 1
                if pos == len(raw_images):
                    pos = 0
                break
            elif key & 0xFF == ord('p'):
                if not affine_dict[pos].isInstance():
                    if not affine_dict[pos].ponit1.isInstance:
                        affine_dict[pos].ponit1.x = x1
                        affine_dict[pos].ponit1.y = y1
                        status_dict[pos] = status.point2
                        if pos == (stack_length - 1):
                            affine_dict[pos].ponit2.x = 1
                            affine_dict[pos].ponit2.y = 1
                            status_dict[pos] = status.vertical_line
                            if vertical_line.isInstance():
                                status_dict[pos] = status.critical_line

                    elif not affine_dict[pos].ponit2.isInstance:
                        affine_dict[pos].ponit2.x = x1
                        affine_dict[pos].ponit2.y = y1
                        status_dict[pos] = status.vertical_line
                        if vertical_line.isInstance():
                            status_dict[pos] = status.critical_line

                elif not vertical_line.isInstance():
                    vertical_line.point1.x = x1
                    vertical_line.point1.y = y1
                    vertical_line.point2.x = x2
                    vertical_line.point2.y = y2
                    vertical_line.pos = pos
                    status_dict[pos] = status.critical_line
                elif not critical_line.isInstance():
                    critical_line.point1.x = x1
                    critical_line.point1.y = y1
                    critical_line.point2.x = x2
                    critical_line.point2.y = y2
                    critical_line.pos = pos

                break
            elif key & 0xFF == ord('q'):
                print("All saved!")
                finshed_flag = True
                break

        for index in range(stack_length):
            try:
                if not affine_dict[index].isInstance():
                    finshed_flag = False
                    break
            except:
                finshed_flag = False
                break
        if not vertical_line.isInstance() or not critical_line.isInstance():
            finshed_flag = False
        if finshed_flag:
            break
    cv2.destroyAllWindows()
    (rows, cols, _) = raw_images[0].shape
    radians = math.atan2(float(vertical_line.point2.y - vertical_line.point1.y),
                                    float(vertical_line.point2.x - vertical_line.point1.x))
    angle = math.degrees(radians) - 90.
    print("angle", angle)
    rotation_matrix = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)

    original_index = critical_line.pos
    original_point = (critical_line.point1.x + critical_line.point2.x, critical_line.point1.y + critical_line.point2.y)
    processed_img = [None for i in range(stack_length)]

    for img_index in range(original_index, -1, -1):
        temp_img = copy.deepcopy(raw_images[img_index])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

        if img_index == original_index:
            diff_x = ((cols - 1) / 4.0) - (float(critical_line.point1.x + critical_line.point2.x) / 2.)
            diff_y = ((rows - 1) / 4.0) - (float(critical_line.point1.y + critical_line.point2.y) / 2.)
            affine_dict[img_index].ponit1.x += diff_x
            affine_dict[img_index].ponit1.y += diff_y
            affine_dict[img_index].ponit2.x += diff_x
            affine_dict[img_index].ponit2.y += diff_y
            shift_matrix = np.float32([
                [1, 0, 2. * diff_x],
                [0, 1, 2. * diff_y]
            ])
            shifted_img = cv2.warpAffine(temp_img, shift_matrix, (cols, rows))

        else:
            diff_x = affine_dict[img_index + 1].ponit1.x - affine_dict[img_index].ponit2.x
            diff_y = affine_dict[img_index + 1].ponit1.y - affine_dict[img_index].ponit2.y
            affine_dict[img_index].ponit1.x += diff_x
            affine_dict[img_index].ponit1.y += diff_y
            shift_matrix = np.float32([
                [1, 0, 2. * diff_x],
                [0, 1, 2. * diff_y]
            ])
            shifted_img = cv2.warpAffine(temp_img, shift_matrix, (cols, rows))

        rotated_img = cv2.warpAffine(shifted_img, rotation_matrix, (cols, rows))
        processed_img[img_index] = rotated_img

    for img_index in range(original_index + 1, stack_length):
        temp_img = copy.deepcopy(raw_images[img_index])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

        # if img_index == original_index:
        #     continue
        #     # diff_x = ((cols - 1) / 4.0) - (float(critical_line.point1.x + critical_line.point2.x) / 2.)
        #     # diff_y = ((rows - 1) / 4.0) - (float(critical_line.point1.y + critical_line.point2.y) / 2.)
        #     # shift_matrix = np.float32([
        #     #     [1, 0, 2. * diff_x],
        #     #     [0, 1, 2. * diff_y]
        #     # ])
        #     # shifted_img = cv2.warpAffine(temp_img, shift_matrix, (cols, rows))
        #
        # else:
        diff_x = affine_dict[img_index - 1].ponit2.x - affine_dict[img_index].ponit1.x
        diff_y = affine_dict[img_index - 1].ponit2.y - affine_dict[img_index].ponit1.y
        affine_dict[img_index].ponit2.x += diff_x
        affine_dict[img_index].ponit2.y += diff_y
        shift_matrix = np.float32([
            [1, 0, 2. * diff_x],
            [0, 1, 2. * diff_y]
        ])
        shifted_img = cv2.warpAffine(temp_img, shift_matrix, (cols, rows))

        rotated_img = cv2.warpAffine(shifted_img, rotation_matrix, (cols, rows))
        processed_img[img_index] = rotated_img

    section_thickness = float(section_thickness) / 1000.
    for i in range(stack_length):
        z_index = float(i - original_index) * section_thickness
        img_name = os.path.basename(input_folder)
        img_name = img_name + ", x0.005464, y0.005464, z%.2f, umm, 009, %f.tif" % (section_thickness, z_index)
        img_dir = os.path.join(output_folder, img_name)
        cv2.imwrite(img_dir, processed_img[i])
