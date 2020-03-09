'''
This script contains the necessary image processing functions in both preprocessing stage and post processing stage.
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab
'''
# The pixel poistion of the original point in allen atlas
ATLAS_CERTER_POSITION = (208, 212, 228)
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
import pickle as pk
import skimage.io as io
from present import show_img, merge_layers, load_img
from bead_finder import save_bead_mask
from ants_utils import quick, apply_transform
import scipy.fftpack as fp
import pandas as pd
from tqdm import tqdm
from preprocess import prepare_tissue_image
import copy
from enum import Enum

# global variables for drawing
drawing = False
x1, y1, x2, y2 = 0, 0, 0, 0

def pixel2mm(point, centre_point, pixel2mm=0.005464):
    '''
    transform the point in Mathew's scale to pixel scale
    :param point: The point you would like to transform
    :param centre_point: The center point of the image
    :param pixel2mm: The ratio for scaling
    :return:
    '''
    mm_x = (float(centre_point[0]) - float(point[0])) * pixel2mm
    mm_y = (float(centre_point[1]) - float(point[1])) * pixel2mm
    return [mm_x, mm_y]

def remove_background(img_frame):
    '''
    Transform the position of the brain in the atlas frame to adapt image frame
    :param img_frame:
    :param atlas_frame:
    :return:
    '''
    # img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    threshold = get_adaptive_threshold(img_frame)
    ret, th = cv2.threshold(img_frame, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    th = cv2.erode(th, kernel, iterations=2)
    _, contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tissue_frame = img_frame.copy()
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 2e5 and cv2.contourArea(contours[i]) < 1e7:
            x_t, y_t, w_t, h_t = cv2.boundingRect(contours[i])
            point_t = (int(x_t + 0.5 * w_t), int(y_t + 0.5 * h_t))
            if is_in_center(point_t, tissue_frame):
                x, y, w, h = cv2.boundingRect(contours[i])
                tissue_frame[:, 0: x] = 0
                tissue_frame[:, x + w:] = 0
                tissue_frame[0:y, :] = 0
                tissue_frame[y + h:, :] = 0
                mask = np.zeros(tissue_frame.shape).astype(np.uint8)
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)
                tissue_frame = cv2.bitwise_and(tissue_frame, tissue_frame, mask=mask)
    return tissue_frame

def read_img(file_path):
    """
    Load an image and convert the color channel of it.
    :param file_path:
    :return:
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def fft(img, frequency_threshold=10, brightness_threshold=40, show=False):
    """
    Perform Fourier transform and also locate the little shinny beads in the image.
    :param img:
    :param frequency_threshold:
    :param brightness_threshold:
    :param show:
    :return:
    """
    F1 = fp.fft2((img).astype(float))
    F2 = fp.fftshift(F1)
    (w, h) = img.shape
    half_w, half_h = int(w / 2), int(h / 2)
    # high pass filter
    n = frequency_threshold
    F2[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0  # select all but the first 50x50 (low) frequencies
    im1 = fp.ifft2(fp.ifftshift(F2)).real

    retval, threshold = cv2.threshold(im1, brightness_threshold, 255, cv2.THRESH_BINARY)

    threshold = threshold.astype('uint8')
    img = img.astype('uint8')
    markers = open_operation(img, threshold)

    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    canvas = np.zeros((img.shape[0], img.shape[1], 3))
    canvas[:, :, 0] = img
    canvas[:, :, 1] = img
    canvas[:, :, 2] = img
    canvas = canvas.astype(np.uint8)
    coor_list = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 400:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if show:
                cv2.drawContours(canvas, c, -1, (0, 255, 0), 1)
                cv2.circle(canvas, (cX, cY), 1, (255, 0, 0), -1)
            if [cX, cY] not in coor_list:
                coor_list.append([cX, cY])
    if show:
        result = [m2, canvas]
        show_imgs(result)
    return coor_list

def open_operation(img, thresh):
    """
    Try to separate the beads.
    :param img:
    :param thresh:
    :return:
    """
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(dist_transform, 2.4, 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    canvas = np.zeros((img.shape[0], img.shape[1], 3))
    canvas[:, :, 0] = img
    canvas[:, :, 1] = img
    canvas[:, :, 2] = img
    canvas = canvas.astype(np.uint8)
    markers = cv2.watershed(canvas, markers)
    return markers

def show_imgs(img_list):
    """
    Show images
    :param img_list:
    :return:
    """
    img_num = len(img_list)
    row = 1
    col = img_num / 2 + 1
    plt.figure(figsize=(20, 16))

    for i in range(img_num):
        plt.subplot(row, col, i + 1)
        plt.imshow(img_list[i], cmap='gray')

    plt.axis('off')
    plt.show()

def locate_beads(img_path_list, output_csv_folder):
    """
    integrated processes for automatically locating the beads in the images.
    :param img_path_list: a list of aligned images
    :param output_csv_folder: the folder for saving csv result.
    :return:
    """
    data_dict = {"Animal ID": [], "Mean": [], "X": [], "Y": [], "Z": [], "Bead Area": [], "Bead Circularity": []}
    def key(item):
        return float(os.path.basename(item).split(',')[-1].strip().replace('.tif', ''))
    img_path_list.sort(key=key)

    for img_path in tqdm(img_path_list):
        animal_id = os.path.basename(img_path).split(',')[0].strip('_')
        z = key(img_path)
        img = read_img(img_path)
        img = remove_background(img)
        xy_list = fft(img, 30, 30)
        for tu in xy_list:
            centre_point = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
            pixel_value = img[tu[1], tu[0]]
            tu = pixel2mm(tu, centre_point)
            tu.append(z)
            data_dict["Animal ID"].append(animal_id)
            data_dict["Mean"].append(pixel_value)
            data_dict["X"].append(tu[0])
            data_dict["Y"].append(tu[1])
            data_dict["Z"].append(tu[2])
            data_dict["Bead Area"].append(0)
            data_dict["Bead Circularity"].append(0)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(os.path.join(output_csv_folder, 'auto_segmentation.csv'), index=False)

def click_and_draw_line(event, x, y, flags, param):
    """
    Painting method in OpenCv
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

def locate_beads_manual(img_path_list, output_csv_folder):
    """
    Method for manual locating the beads
    :param img_path_list: a list of aligned images.
    :param output_csv_folder: a folder for saving output csv files.
    :return:
    """
    global x1, y1, x2, y2, drawing
    data_dict = {"Animal ID": [], "Mean": [], "X": [], "Y": [], "Z": [], "Bead Area":[], "Bead Circularity":[]}
    def key(item):
        return float(os.path.basename(item).split(',')[-1].strip().replace('.tif', ''))
    img_path_list.sort(key=key)

    class Enhanced_image:
        def __init__(self, animal_id, z_key, image, bead_list=[]):
            self.animal_id = animal_id
            self.z_key = z_key
            self.image = image
            self.bead_list = bead_list

    class Bead:
        def __init__(self, x, y, scale):
            assert scale != 0
            self._x = float(x) / float(scale)
            self._y = float(y) / float(scale)
            self.scale = scale
        @property
        def x(self):
            return int(self._x)

        @property
        def y(self):
            return int(self._y)

        @property
        def x_show(self):
            return int(self._x * self.scale)

        @property
        def y_show(self):
            return int(self._y * self.scale)

    class Brush:
        Color_labeled = (0, 255, 0)
        Color_unlabeled = (255, 0, 0)
        Point_size_labeled = 5
        Point_size_unlabeled = 3

    enhanced_image_list = []
    cv2.namedWindow("Label the beads")
    cv2.setMouseCallback("Label the beads", click_and_draw_line)
    for img_path in tqdm(img_path_list):
        animal_id = os.path.basename(img_path).split(',')[0].strip('_')
        z = key(img_path)
        img = read_img(img_path)
        img = remove_background(img)
        enhanced_image_list.append(Enhanced_image(animal_id, z, img, []))


    viewing_flag = True
    viewing_pos = 0
    scale = 0.5
    while viewing_flag:

        while True:
            show_img = copy.deepcopy(enhanced_image_list[viewing_pos].image)
            show_img = cv2.resize(show_img, (int(show_img.shape[1] * scale), int(show_img.shape[0] * scale)))
            for bead in enhanced_image_list[viewing_pos].bead_list:
                cv2.line(show_img, (bead.x_show, bead.y_show), (bead.x_show, bead.y_show), Brush.Color_labeled,
                         thickness=Brush.Point_size_labeled)


            cv2.putText(show_img, "Pos: %3d" % viewing_pos, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(show_img, "Press S to save current point", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(show_img, "Press E to erase last point", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(show_img, "Press Q to quit and save all", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(show_img, "Others: Press A and D to go previous or next", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.line(show_img, (x1, y1), (x1, y1), Brush.Color_unlabeled, thickness=Brush.Point_size_unlabeled)
            cv2.imshow("Label the beads", show_img)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('a'):
                viewing_pos -= 1
                if viewing_pos == -1:
                    viewing_pos = len(enhanced_image_list) - 1
                break
            elif key == ord('d'):
                viewing_pos += 1
                if viewing_pos == len(enhanced_image_list):
                    viewing_pos = 0
                break
            elif key == ord('e'):
                if len(enhanced_image_list[viewing_pos].bead_list) > 0:
                    enhanced_image_list[viewing_pos].bead_list.pop()
                break
            elif key == ord('s'):
                enhanced_image_list[viewing_pos].bead_list.append(Bead(x1, y1, scale=scale))
                break
            elif key == ord('q'):
                viewing_flag = False
                break
    cv2.destroyAllWindows()
    centre_point = (int(enhanced_image_list[0].image.shape[1] * 0.5), int(enhanced_image_list[0].image.shape[0] * 0.5))
    for enhanced_img in enhanced_image_list:
        for bead in enhanced_img.bead_list:
            pixel_value = enhanced_img.image[bead.y, bead.x]
            triple = pixel2mm((bead.x, bead.y), centre_point)
            triple.append(enhanced_img.z_key)
            data_dict["Animal ID"].append(enhanced_img.animal_id)
            data_dict["Mean"].append(pixel_value)
            data_dict["X"].append(triple[0])
            data_dict["Y"].append(triple[1])
            data_dict["Z"].append(triple[2])
            data_dict["Bead Area"].append(0)
            data_dict["Bead Circularity"].append(0)

    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(os.path.join(output_csv_folder, 'manual_segmentation.csv'), index=False)

def segment_bead_csv(root_dir, auto=True):

    img_root_dir = os.path.join(root_dir, "processed")
    img_path_list = []
    out_put_folder = os.path.join(root_dir, "data")
    for img_dir in os.listdir(img_root_dir):
        if img_dir.endswith(".tif"):
            img_path_list.append(os.path.join(img_root_dir, img_dir))
    if auto:
        print("Computing segmentation using Fourier transform...")
        if not os.path.exists(os.path.join(out_put_folder, 'auto_segmentation.csv')):
            locate_beads(img_path_list, out_put_folder)
    else:
        if not os.path.exists(os.path.join(out_put_folder, 'manual_segmentation.csv')):
            locate_beads_manual(img_path_list, out_put_folder)

def is_in_center(centerPonit, real_img):
    '''
    Given a center point and the image containing the brain, determine whether the region is in the cernter of image or not
    :param centerPonit:
    :param real_img:
    :return: boolean
    '''
    x = float(real_img.shape[1] * 0.5)
    y = float(real_img.shape[0] * 0.5)
    if (centerPonit[0] > (x * 0.6)) and (centerPonit[0] < (x * 1.4)) and (centerPonit[1] > (y * 0.4)) and (centerPonit[1] < (y * 1.2)):
        return True
    else:
        return False

def get_pure_brain_atlas(atlas_frame, refactored_atlas_center):
    '''
    Return only the brain part in the atlas frame, and remove the outside black area
    :param atlas_frame:
    :param refactored_atlas_center:
    :return:
    '''
    (height, width) = atlas_frame.shape
    row1 = 0
    col1 = 0
    row2 = 0
    col2 = 0
    for row in range(0, height):
        if np.asarray(atlas_frame[row, :]).sum() > 0:
            row1 = row
            break

    for row in range(height - 1, 0, -1):
        if np.asarray(atlas_frame[row, :]).sum() > 0:
            row2 = row
            break

    for col in range(0, width):
        if np.asarray(atlas_frame[:, col]).sum() > 0:
            col1 = col
            break

    for col in range(width - 1, 0, -1):
        if np.asarray(atlas_frame[:, col]).sum() > 0:
            col2 = col
            break

    atlas_frame = atlas_frame[row1:row2, col1:col2]
    atlas_center = (refactored_atlas_center[0] - col1, refactored_atlas_center[1] - row1)

    return atlas_frame, atlas_center

def get_adaptive_threshold(img_gray, show=False):
    '''
    Using the hist diagram to calculate the adaptive threshold of binarizing th image
    :param img_gray: single channel gray image
    :param show: if show is true, it will open a window containing the hist diagram
    :return: Adapative threshold value
    '''
    hist_full = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    if show:
        plt.plot(hist_full)
        plt.show()
    hist_ave = sum(hist_full[1:]) / 255.
    window_size = 5
    for i in range(10, 50):
        temp = hist_full[i: i + window_size].reshape((window_size  , ))
        if np.gradient(temp).max() < 0 and (temp.sum() / float(window_size)) < hist_ave:
            return i
    return 25

def preprocess_pair(img_frame, atlas_frame, ann_frame, show=False):
    '''
    Transform the position of the brain in the atlas frame to adapt image frame
    :param img_frame:
    :param atlas_frame:
    :return:
    '''
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    threshold = get_adaptive_threshold(img_frame)
    ret, th = cv2.threshold(img_frame, threshold, 255, cv2.THRESH_BINARY)


    kernel = np.ones((7, 7), np.uint8)
    th = cv2.erode(th, kernel, iterations=2)
    _, contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tissue_frame = img_frame.copy()

    # show_img(th, False)

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 2e5 and cv2.contourArea(contours[i]) < 5e6:
            x_t, y_t, w_t, h_t = cv2.boundingRect(contours[i])
            point_t = (int(x_t + 0.5 * w_t), int(y_t + 0.5 * h_t))

            if is_in_center(point_t, tissue_frame):
                x, y, w, h = cv2.boundingRect(contours[i])
                tissue_frame[:, 0: x] = 0
                tissue_frame[:, x + w:] = 0
                tissue_frame[0:y, :] = 0
                tissue_frame[y + h:, :] = 0
                mask = np.zeros(tissue_frame.shape).astype(np.uint8)
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)
                tissue_frame = cv2.bitwise_and(tissue_frame, tissue_frame, mask=mask)
                if show:
                    print("normal")
                    cv2.rectangle(tissue_frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
                    show_img(tissue_frame, False)

    # show_tissue_frame = cv2.resize(tissue_frame, (int(tissue_frame.shape[1] * 0.5), int(tissue_frame.shape[0] * 0.5)))
    # cv2.imshow('tissue', show_tissue_frame)
    # cv2.waitKey()

    cur_h = 0
    cur_w = 0
    (height, length) = atlas_frame.shape

    # show_img(atlas_frame, False)

    for row in range(height):
        if np.asarray(atlas_frame[row, :]).sum() > 0:
            cur_h += 1
    for col in range(length):
        if np.asarray(atlas_frame[:, col]).sum() > 0:
            cur_w += 1
    #
    # print(w, cur_w)
    # print(h, cur_h)

    try:
        w_factor = float(w) / float(cur_w)
        h_factor = float(h) / float(cur_h)
    except:
        tissue_frame = img_frame.copy()

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(tissue_frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
            print("exception")
            show_img(tissue_frame, False)

    # print(w_factor, h_factor)

    refactored_atlas_center = (int(ATLAS_CERTER_POSITION[2] * w_factor), int(ATLAS_CERTER_POSITION[1] * h_factor))

    atlas_frame = cv2.resize(atlas_frame, (int(atlas_frame.shape[1] * w_factor), int(atlas_frame.shape[0] * h_factor)), interpolation=cv2.INTER_NEAREST)
    ann_frame = cv2.resize(ann_frame, (int(ann_frame.shape[1] * w_factor), int(ann_frame.shape[0] * h_factor)), interpolation=cv2.INTER_NEAREST)

    # print(atlas_frame.shape)

    atlas_frame, atlas_center = get_pure_brain_atlas(atlas_frame, refactored_atlas_center)
    ann_frame, _ = get_pure_brain_atlas(ann_frame, refactored_atlas_center)

    # print(atlas_frame.shape)
    # print("===========")
    # show_img(atlas_frame, False)

    canvas_atlas = np.zeros((img_frame.shape[0], img_frame.shape[1])).astype(np.uint8)
    canvas_ann = np.zeros((img_frame.shape[0], img_frame.shape[1])).astype(np.uint16)

    atlas_size = atlas_frame.shape
    canvas_atlas[0:atlas_size[0], 0:atlas_size[1]] = atlas_frame
    ann_size = ann_frame.shape
    canvas_ann[0:ann_size[0], 0:ann_size[1]] = ann_frame

    canvas_center = (int(canvas_atlas.shape[1] * 0.5), int(canvas_atlas.shape[0] * 0.5))

    shift_col =  canvas_center[0] - atlas_center[0]
    shift_row= canvas_center[1] - atlas_center[1]
    M = np.float32([[1, 0, shift_col], [0, 1, shift_row]])

    canvas_atlas = cv2.warpAffine(canvas_atlas, M, (canvas_atlas.shape[1], canvas_atlas.shape[0]))
    canvas_ann = cv2.warpAffine(canvas_ann, M, (canvas_ann.shape[1], canvas_ann.shape[0]))

    return tissue_frame, canvas_atlas, canvas_ann

def calculate_shift(img_dir):
    '''
    Calculate the index shift between image and nrrd
    :param img_dir:
    :return:
    '''
    def z_key(elem):
        return -float(elem.split(',')[-1].strip().split('.tif')[0])
    tif_list = os.listdir(img_dir)
    tif_list.sort(key=z_key)
    i = 0
    for tif_name in tif_list:
        last_temp = tif_name.split(',')[-1].strip().split('.tif')[0]
        if float(last_temp) == 0:
            index = i
            break
        i += 1
    shift = ATLAS_CERTER_POSITION[0] - (index * 2)
    return shift

def save_pair_images(img_dir, save_dir="/home/silasi/ants_data/name"):
    '''
    save the pair of images in to specific folder.
    :param img_dir:
    :param save_dir:
    :return:
    '''

    def z_key(elem):
        return -float(elem.split(',')[-1].strip().split('.tif')[0])
    tif_list = os.listdir(img_dir)
    tif_list.sort(key=z_key)

    f1 = open(os.path.join("..", "atlas_reference", "atlas_plot.pickle"), 'rb')
    atlas_plot_data = pk.load(f1)
    f2 = open(os.path.join("..", "atlas_reference", "atlas_dict.pickle"), 'rb')
    atlas_dict_data = pk.load(f2)
    shift = calculate_shift(img_dir)

    print("Calculating........")
    save_dir_atlas = os.path.join(save_dir, 'atlas')
    save_dir_ann = os.path.join(save_dir, 'ann')
    save_dir_tissue = os.path.join(save_dir, 'tissue')
    if not os.path.exists(save_dir_atlas):
        os.mkdir(save_dir_atlas)
    if not os.path.exists(save_dir_tissue):
        os.mkdir(save_dir_tissue)
    if not os.path.exists(save_dir_ann):
        os.mkdir(save_dir_ann)

    for i in tqdm(range(len(tif_list))):
        tif_name = tif_list[i]
        img_frame = cv2.imread(os.path.join(img_dir, tif_name))

        atlas_frame = atlas_plot_data[i * 2 + shift]
        ann_frame = atlas_dict_data[i * 2 + shift]
        ann_frame = np.asarray(ann_frame, dtype=np.uint16)

        tissue_frame, canvas_atlas, canvas_ann = preprocess_pair(img_frame, atlas_frame, ann_frame, False)

        canvas_atlas = cv2.resize(canvas_atlas, (int(canvas_atlas.shape[1] * 0.5), int(canvas_atlas.shape[0] * 0.5)))
        imsave(os.path.join(save_dir_atlas, '%d.tif' % i), canvas_atlas)

        canvas_ann = cv2.resize(canvas_ann, (int(canvas_ann.shape[1] * 0.5), int(canvas_ann.shape[0] * 0.5)), interpolation=cv2.INTER_NEAREST)
        np.save(os.path.join(save_dir_ann, '%d.npy' % i), canvas_ann)

        tissue_frame = cv2.resize(tissue_frame, (int(tissue_frame.shape[1] * 0.5), int(tissue_frame.shape[0] * 0.5)))
        imsave(os.path.join(save_dir_tissue, '%d.tif' % i), tissue_frame)



def prepare_atlas():
    """
    Transform the mhd as well as the raw image file into pickle files and also rotate into right direction.
    :return:
    """
    img = io.imread('..' + os.sep + 'atlas_reference' + os.sep + 'atlasVolume.mhd', plugin='simpleitk')
    annotation = io.imread('..' + os.sep + 'atlas_reference' + os.sep + 'annotation.mhd', plugin='simpleitk')
    assert img.shape == annotation.shape, "Image dose not match the annotation file!"
    atlas_list = []
    ann_list = []
    for i in range(img.shape[2]):
        img90 = np.rot90(img[:, :, i], k=-1)
        img90 = np.asarray(img90)
        atlas_list.append(img90)

        ann90 = np.rot90(annotation[:, :, i], k=-1)
        ann90 = np.asarray(ann90)
        ann_list.append(ann90)

    f1 = open(".." + os.sep + "atlas_reference" + os.sep + "atlas_plot.pickle", 'wb')
    pk.dump(atlas_list, f1)
    f2 = open(".." + os.sep + "atlas_reference" + os.sep + "atlas_dict.pickle", 'wb')
    pk.dump(ann_list, f2)

def summary_single_section(dictionary, mask, annotation):
    if dictionary is None:
        dictionary = {}
    mask = np.asarray(mask, dtype=np.uint16)
    result = cv2.bitwise_and(annotation, mask)
    result_list = np.where(result > 0)
    x_list = result_list[0].tolist()
    y_list = result_list[1].tolist()

    for (x, y) in zip(x_list, y_list):
        pixel_value = annotation[x][y]

        if not pixel_value in dictionary.keys():
            dictionary[pixel_value] = 0
        dictionary[pixel_value] += 1
    return dictionary

def check_create_dirs(save_dir):
    """
    Used to check and organize the directory structure
    :param save_dir:
    :return:
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_atlas = os.path.join(save_dir, 'atlas')
    save_dir_ann = os.path.join(save_dir, 'ann')
    save_dir_tissue = os.path.join(save_dir, 'tissue')
    if not os.path.exists(save_dir_atlas):
        os.mkdir(save_dir_atlas)
    if not os.path.exists(save_dir_tissue):
        os.mkdir(save_dir_tissue)
    if not os.path.exists(save_dir_ann):
        os.mkdir(save_dir_ann)
    if not os.path.exists(os.path.join(save_dir, "post_bead")):
        os.mkdir(os.path.join(save_dir, "post_bead"))
    if not os.path.exists(os.path.join(save_dir, "post_tissue")):
        os.mkdir(os.path.join(save_dir, "post_tissue"))
    if not os.path.exists(os.path.join(save_dir, "bead")):
        os.mkdir(os.path.join(save_dir, "bead"))
    if not os.path.exists(os.path.join(save_dir, 'output')):
        os.mkdir(os.path.join(save_dir, 'output'))

def get_query_dict():
    """
    Load query dictionary indicated by id.
    :return:
    """
    query_path = os.path.join("atlas_reference", "query.csv")
    query_path = os.path.abspath(query_path)
    query_path = query_path.replace("/src", "")
    query_path = os.path.normpath(query_path)
    df = pd.read_csv(query_path)
    query_dict = df.to_dict()
    processed_dict = {}
    keys = list(query_dict.keys())
    keys.remove('id')
    id_keys = list(query_dict['id'].keys())
    id_keys.sort()
    for id_key in id_keys:
        id = query_dict['id'][id_key]
        processed_dict[id] = {}
        for key in keys:
            processed_dict[id][key] = query_dict[key][id_key]
    query_dict = processed_dict
    return query_dict

def main(root_dir, save_dir, prepare_atlas_tissue=False, registration=False,
         Ants_script="/home/silasi/ANTs/Scripts",
         app_tran=False, write_summary=False, show=False,
         show_atlas=False, intro=True, auto_seg=True):
    """
    Show function is not compatible with writing csv funtion. Do one thing at a time.
    :param root_dir: Directory for handling all animal folders you would like to analyse.
    :param save_dir: Directory to save the analysed result.
    :param prepare_atlas_tissue: True for the first time to run on this animal. Set fault to save time after that.
    :param registration: Default True, use ANTs to align brain. Set as fause in the second time to save time.
    :param app_tran: Always true.
    :param write_summary: True for generating summary csv and tree graph.
    :param show: True for show in opencv frame, cannot be true when write summary.
    :return:
    """

    assert not (show and write_summary), "Show function is not compatible with sumarry function"
    name_list = os.listdir(root_dir)
    for name in name_list:
        img_dir = os.path.join(root_dir, name, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
        save_directory = os.path.join(save_dir, name)
        check_create_dirs(save_directory)

        segment_bead_csv(os.path.join(root_dir, name), auto=auto_seg)
        save_bead_mask(save_directory, os.path.join(root_dir, name), show_circle=show, auto=auto_seg)

        if prepare_atlas_tissue:
            prepare_atlas()
            save_pair_images(img_dir, save_dir=save_directory)

        result_dict = None
        length = len(os.listdir(os.path.join(save_directory, 'atlas')))
        for i in tqdm(range(length)):
            atlas_dir = os.path.join(save_directory, 'atlas' + os.sep + '%d.tif' % i)
            tissue_dir = os.path.join(save_directory, 'tissue' + os.sep + '%d.tif' % i)

            output_dir = os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i)

            if registration:
                quick(atlas_dir, tissue_dir, output_dir, ANTs_script=Ants_script)

            transforms = [os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '0GenericAffine.mat'),
                          os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '1Warp.nii.gz')]
            bead_dir = os.path.join(save_directory, 'bead' + os.sep + '%d.tif' % i)

            if app_tran:
                apply_transform(bead_dir, atlas_dir, transforms, os.path.join(save_directory, "post_bead" + os.sep + "%d.nii" % i))
                apply_transform(tissue_dir, atlas_dir, transforms, os.path.join(save_directory, "post_tissue" + os.sep + "%d.nii"%i))

            if write_summary and not show:
                bead = load_img(os.path.join(save_directory, "post_bead", "%d.nii"%i), 'nii')
                ann = np.load(os.path.join(save_directory, "ann", "%d.npy" % i))
                result_dict = summary_single_section(result_dict, bead, ann)

        if write_summary and not show:
            query_dict = get_query_dict()
            csv_dict = {"label": [], "text_label": [], "number": [],
                        "structure_id_path": [], "percentage": []}

            total_bead_number = 0.
            not_found = 0

            for key in result_dict:
                total_bead_number += result_dict[key]

            for key in result_dict:
                if key in query_dict.keys():
                    csv_dict["label"].append(key)
                    csv_dict["text_label"].append(query_dict[key]['name'])
                    csv_dict["structure_id_path"].append(query_dict[key]['structure_id_path'])
                    csv_dict["number"].append(result_dict[key])
                    csv_dict["percentage"].append(float(result_dict[key]) / total_bead_number)
                else:
                    not_found += result_dict[key]
                    print("ID:" + str(key) + " Not found, containing bead: %d" % result_dict[key])

            draw_tree_graph(csv_dict, save_directory, query_dict)

            csv_dict["label"].append("nan")
            csv_dict["text_label"].append("Background")
            csv_dict["structure_id_path"].append("nan")
            csv_dict["number"].append(not_found)
            csv_dict["percentage"].append(float(not_found) / total_bead_number)
            df = pd.DataFrame(csv_dict)
            df.to_csv(os.path.join(save_directory, "summary.csv"))

        if show:
            if not show_atlas:
                merge_layers(name, save_dir, 'nii', 'nii', 'npy', intro)
            else:
                merge_layers(name, save_dir, 'nii', 'nii', 'tif', intro)

def draw_tree_graph(csv_dict, save_directory, query_dict):
    """
    Only utilize it when you want to generate a tree graph.
    :param csv_dict:
    :param save_directory:
    :param query_dict:
    :return:
    """
    from anytree import Node, RenderTree, AsciiStyle, LevelOrderIter
    from anytree.exporter import DotExporter
    tree_path_list = csv_dict["structure_id_path"]
    count_list = csv_dict["number"]
    label_list = csv_dict["label"]
    percentage_list = csv_dict["percentage"]
    node_dict = {}
    root_node = None

    for path_index in range(len(tree_path_list)):
        tree_path = tree_path_list[path_index]
        node_list = [int(item) for item in tree_path.split('/')[1: -1]]
        for i in range(len(node_list)):
            node_id = node_list[i]
            if node_id not in node_dict.keys():
                if node_id in query_dict.keys():
                    node_name = query_dict[node_id]['name']
                else:
                    node_name = str(node_id)

                if i == len(node_list) - 1:
                    node_dict[node_id] = Node(node_name, parent=node_dict[node_list[i - 1]],
                                              count=count_list[path_index],
                                              percentage=percentage_list[path_index])
                else:
                    if i == 0:
                        root_node = Node(node_name, count=0, percentage=.0)
                        node_dict[node_id] = root_node
                    else:
                        node_dict[node_id] = Node(node_name, parent=node_dict[node_list[i - 1]],
                                                  count=0, percentage=.0)
                    if node_id in label_list:
                        Node(node_name + "_other", parent=node_dict[node_id],
                             count=count_list[label_list.index(node_id)],
                             percentage=percentage_list[label_list.index(node_id)])

    node_level_order_list = list(LevelOrderIter(root_node))
    for i in range(len(node_level_order_list) - 1, 0, -1):
        node_level_order_list[i].parent.count += node_level_order_list[i].count
        node_level_order_list[i].parent.percentage += node_level_order_list[i].percentage


    def nodenamefunc(node):
        return '%s, beads number:%2d, beads percentage: %.5f %%' % (node.name, node.count, node.percentage*100.)
    DotExporter(root_node, nodenamefunc=nodenamefunc).to_picture(os.path.join(save_directory, "tree.png"))
    with open(os.path.join(save_directory, 'tree.txt'), 'w') as f:
        for pre, _, node in RenderTree(root_node):
            f.write('%s:%s, beads number:%2d, beads percentage: %.5f %% \n' % (pre, node.name, node.count, node.percentage*100.))

# def run_one_brain(brain_dir, save_dir, prepare_atlas_tissue=False, registration=False,
#          Ants_script="/home/silasi/ANTs/Scripts",
#          app_tran=False, write_summary=False, show=False,
#          show_atlas=False, intro=True, auto_seg=True):
#     """
#     Show function is not compatible with writing csv funtion. Do one thing at a time.
#     :param brain_dir: Directory for handling the brain you would like to analyse.
#     :param save_dir: Directory to save the analysed result.
#     :param prepare_atlas_tissue: True for the first time to run on this animal. Set fault to save time after that.
#     :param registration: Default True, use ANTs to align brain. Set as fause in the second time to save time.
#     :param app_tran: Always true.
#     :param write_summary: True for generating summary csv and tree graph.
#     :param show: True for show in opencv frame, cannot be true when write summary.
#     :return:
#     """
#     brain_name = os.path.basename(brain_dir)
#     assert not (show and write_summary), "Show function is not compatible with sumarry function"
#
#     img_dir = os.path.join(brain_dir, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
#     save_directory = os.path.join(save_dir, brain_name)
#     check_create_dirs(save_directory)
#     if auto_seg:
#         segment_bead_csv(os.path.join(brain_dir))
#     save_bead_mask(save_directory, os.path.join(brain_dir), show_circle=show, auto=auto_seg)
#     if prepare_atlas_tissue:
#         prepare_atlas()
#         save_pair_images(img_dir, save_dir=save_directory)
#
#     result_dict = None
#     length = len(os.listdir(os.path.join(save_directory, 'atlas')))
#     for i in tqdm(range(length)):
#         atlas_dir = os.path.join(save_directory, 'atlas' + os.sep + '%d.tif' % i)
#         tissue_dir = os.path.join(save_directory, 'tissue' + os.sep + '%d.tif' % i)
#
#         output_dir = os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i)
#
#         if registration:
#             quick(atlas_dir, tissue_dir, output_dir, ANTs_script=Ants_script)
#
#         transforms = [os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '0GenericAffine.mat'),
#                       os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '1Warp.nii.gz')]
#         bead_dir = os.path.join(save_directory, 'bead' + os.sep + '%d.tif' % i)
#
#         if app_tran:
#             apply_transform(bead_dir, atlas_dir, transforms, os.path.join(save_directory, "post_bead" + os.sep + "%d.nii" % i))
#             apply_transform(tissue_dir, atlas_dir, transforms, os.path.join(save_directory, "post_tissue" + os.sep + "%d.nii"%i))
#
#         if write_summary and not show:
#             bead = load_img(os.path.join(save_directory, "post_bead", "%d.nii"%i), 'nii')
#             ann = np.load(os.path.join(save_directory, "ann", "%d.npy" % i))
#             result_dict = summary_single_section(result_dict, bead, ann)
#
#     if write_summary and not show:
#         query_dict = get_query_dict()
#         csv_dict = {"label": [], "text_label": [], "number": [],
#                     "structure_id_path": [], "percentage": []}
#
#         total_bead_number = 0.
#         not_found = 0
#
#         for key in result_dict:
#             total_bead_number += result_dict[key]
#
#         for key in result_dict:
#             if key in query_dict.keys():
#                 csv_dict["label"].append(key)
#                 csv_dict["text_label"].append(query_dict[key]['name'])
#                 csv_dict["structure_id_path"].append(query_dict[key]['structure_id_path'])
#                 csv_dict["number"].append(result_dict[key])
#                 csv_dict["percentage"].append(float(result_dict[key]) / total_bead_number)
#             else:
#                 not_found += result_dict[key]
#                 print("ID:" + str(key) + " Not found, containing bead: %d" % result_dict[key])
#
#         draw_tree_graph(csv_dict, save_directory, query_dict)
#
#         csv_dict["label"].append("nan")
#         csv_dict["text_label"].append("Background")
#         csv_dict["structure_id_path"].append("nan")
#         csv_dict["number"].append(not_found)
#         csv_dict["percentage"].append(float(not_found) / total_bead_number)
#         df = pd.DataFrame(csv_dict)
#         df.to_csv(os.path.join(save_directory, "summary.csv"))
#
#     if show:
#         if not show_atlas:
#             merge_layers(brain_name, save_dir, 'nii', 'nii', 'npy', intro)
#         else:
#             merge_layers(brain_name, save_dir, 'nii', 'nii', 'tif', intro)

def run_one_brain(brain_dir, save_dir, prepare_atlas_tissue=False, registration=False,
         Ants_script="/home/silasi/ANTs/Scripts",
         app_tran=False, write_summary=False, show=False,
         show_atlas=False, intro=True, auto_seg=True):
    """
    Show function is not compatible with writing csv funtion. Do one thing at a time.
    :param brain_dir: Directory for handling the brain you would like to analyse.
    :param save_dir: Directory to save the analysed result.
    :param prepare_atlas_tissue: True for the first time to run on this animal. Set fault to save time after that.
    :param registration: Default True, use ANTs to align brain. Set as fause in the second time to save time.
    :param app_tran: Always true.
    :param write_summary: True for generating summary csv and tree graph.
    :param show: True for show in opencv frame, cannot be true when write summary.
    :return:
    """
    brain_name = os.path.basename(brain_dir)
    assert not (show and write_summary), "Show function is not compatible with sumarry function"
    raw_img_dir = os.path.join(brain_dir, 'raw')
    assert os.path.exists(raw_img_dir)
    img_dir = os.path.join(brain_dir, 'processed')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    data_dir = os.path.join(brain_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if len(os.listdir(img_dir)) == 0:
        prepare_tissue_image(raw_img_dir, img_dir)

    save_directory = os.path.join(save_dir, brain_name)
    check_create_dirs(save_directory)

    segment_bead_csv(brain_dir, auto=auto_seg)

    save_bead_mask(save_directory, os.path.join(brain_dir), show_circle=show, auto=auto_seg)
    if prepare_atlas_tissue:
        prepare_atlas()
        save_pair_images(img_dir, save_dir=save_directory)

    result_dict = None
    length = len(os.listdir(os.path.join(save_directory, 'atlas')))
    for i in tqdm(range(length)):
        atlas_dir = os.path.join(save_directory, 'atlas' + os.sep + '%d.tif' % i)
        tissue_dir = os.path.join(save_directory, 'tissue' + os.sep + '%d.tif' % i)
        output_dir = os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i)

        if registration:
            quick(atlas_dir, tissue_dir, output_dir, ANTs_script=Ants_script)

        transforms = [os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '0GenericAffine.mat'),
                      os.path.join(save_directory, 'output' + os.sep + 'output_%d_' % i + '1Warp.nii.gz')]
        bead_dir = os.path.join(save_directory, 'bead' + os.sep + '%d.tif' % i)

        if app_tran:
            apply_transform(bead_dir, atlas_dir, transforms, os.path.join(save_directory, "post_bead" + os.sep + "%d.nii" % i))
            apply_transform(tissue_dir, atlas_dir, transforms, os.path.join(save_directory, "post_tissue" + os.sep + "%d.nii"%i))

        if write_summary and not show:
            bead = load_img(os.path.join(save_directory, "post_bead", "%d.nii"%i), 'nii')
            ann = np.load(os.path.join(save_directory, "ann", "%d.npy" % i))
            result_dict = summary_single_section(result_dict, bead, ann)

    if write_summary and not show:
        query_dict = get_query_dict()
        csv_dict = {"label": [], "text_label": [], "number": [],
                    "structure_id_path": [], "percentage": []}

        total_bead_number = 0.
        not_found = 0

        for key in result_dict:
            total_bead_number += result_dict[key]

        for key in result_dict:
            if key in query_dict.keys():
                csv_dict["label"].append(key)
                csv_dict["text_label"].append(query_dict[key]['name'])
                csv_dict["structure_id_path"].append(query_dict[key]['structure_id_path'])
                csv_dict["number"].append(result_dict[key])
                csv_dict["percentage"].append(float(result_dict[key]) / total_bead_number)
            else:
                not_found += result_dict[key]
                print("ID:" + str(key) + " Not found, containing bead: %d" % result_dict[key])
        if len(result_dict.keys()) > 0:
            draw_tree_graph(csv_dict, save_directory, query_dict)

        csv_dict["label"].append("nan")
        csv_dict["text_label"].append("Background")
        csv_dict["structure_id_path"].append("nan")
        csv_dict["number"].append(not_found)
        csv_dict["percentage"].append(float(not_found) / total_bead_number)
        df = pd.DataFrame(csv_dict)
        df.to_csv(os.path.join(save_directory, "summary.csv"))

    if show:
        if not show_atlas:
            merge_layers(brain_name, save_dir, 'nii', 'nii', 'npy', intro)
        else:
            merge_layers(brain_name, save_dir, 'nii', 'nii', 'tif', intro)
if __name__ == '__main__':
    pass

