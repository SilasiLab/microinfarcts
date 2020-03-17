from src.image_processing_utils import *
from src.bead_finder import *
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack as fp
import pandas as pd
from tqdm import tqdm

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
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def fft(img, frequency_threshold=10, brightness_threshold=40, show=False):
    F1 = fp.fft2((img).astype(float))
    F2 = fp.fftshift(F1)
    (w, h) = img.shape
    half_w, half_h = int(w / 2), int(h / 2)
    # high pass filter
    n = frequency_threshold
    F2[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0  # select all but the first 50x50 (low) frequencies
    im1 = fp.ifft2(fp.ifftshift(F2)).real
    # im1 = im1.astype('uint8')
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
    data_dict = {"Animal ID": [], "Mean": [], "X": [], "Y": [], "Z": [], "Bead Area":[], "Bead Circularity":[]}
    def key(item):
        return float(os.path.basename(item).split(',')[-1].strip().replace('.tif', ''))
    img_path_list.sort(key=key)

    for img_path in tqdm(img_path_list):
        animal_id = os.path.basename(img_path).split(',')[0].strip('_')
        z = key(img_path)
        img = read_img(img_path)
        img = remove_background(img)
        xy_list = fft(img, 30, 30, show=False)
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

def segment_bead_csv(root_dir):
    print("Computing segmentation using Fourier transform...")
    img_root_dir = os.path.join(root_dir, "3 - Processed Images/7 - Counted Reoriented Stacks Renamed")
    img_path_list = []
    out_put_folder = os.path.join(root_dir, "5 - Data")
    for img_dir in os.listdir(img_root_dir):
        if img_dir.endswith(".tif"):
            img_path_list.append(os.path.join(img_root_dir, img_dir))
    locate_beads(img_path_list, out_put_folder)

# def get_layer_dictionary_test(csv_path):
#     data_frame = load_data(csv_path)
#     bead_list = find_real_bead(data_frame, deliation_factor=3, ignore_disconnected=1, tolerance=0.3)
#     rootDir = "/mnt/4T/brain_data/cage2-9/"
#     img_dir = os.path.join(rootDir, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
#     origin = calculate_origin(img_dir)
#     def depth_key(elem):
#         return elem.end_z
#     bead_list.sort(key=depth_key)
#     layer_dict = {}
#     for bead in bead_list:
#         pixel_corrdinates = mm2pixel(bead.pos[0], origin)
#         if bead.start_z not in layer_dict.keys():
#             layer_dict[bead.start_z] = [pixel_corrdinates]
#         else:
#             layer_dict[bead.start_z].append(pixel_corrdinates)
#     return layer_dict
#
# def comparision(csv_path_1, csv_path_2, img_path_list):
#     layer_dict_1 = get_layer_dictionary_test(csv_path_1)
#     layer_dict_2 = get_layer_dictionary_test(csv_path_2)
#     sum1, sum2 = 0, 0
#     for key in layer_dict_1.keys():
#         sum1 += len(layer_dict_1[key])
#     for key in layer_dict_2.keys():
#         sum2 += len(layer_dict_2[key])
#     print(sum1, sum2)
#     print("OK")
#     #################################################################################################
#     _rootDir = "/mnt/4T/brain_imgs/73594-2/"
#     _img_dir = os.path.join(_rootDir, "3 - Processed Images", "7 - Counted Reoriented Stacks Renamed")
#     origin = calculate_origin(_img_dir)
#     def load_click_pos(csv_path):
#         data_frame = load_data(csv_path)
#         data_dict = {}
#         for entry in data_frame:
#             if entry[2] not in data_dict.keys():
#                 data_dict[entry[2]] = []
#             pixel_corrdinates = mm2pixel((entry[0], entry[1]), origin)
#             data_dict[entry[2]].append((pixel_corrdinates[0], pixel_corrdinates[1]))
#         return data_dict
#     click_dict_1 = load_click_pos(csv_path_1)
#     click_dict_2 = load_click_pos(csv_path_2)
#     #################################################################################################
#     def key(item):
#         return float(os.path.basename(item).split(',')[-1].strip().replace('.tif', ''))
#     img_path_list.sort(key=key)
#     for img_path in img_path_list:
#         index = key(img_path)
#         print(img_path)
#         canvas = read_img(img_path)
#         canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
#         point_list_1 = []
#         point_list_2 = []
#         if index in click_dict_1.keys():
#             point_list_1 = click_dict_1[index]
#         if index in click_dict_2.keys():
#             point_list_2 = click_dict_2[index]
#         for point in point_list_1:
#             cv2.circle(canvas, point, 1, (255, 0, 0), -1)
#         for point in point_list_2:
#             cv2.circle(canvas, point, 1, (0, 255, 0), -1)
#         show_imgs([canvas])



if __name__ == '__main__':
    root_dir = "/mnt/4T/brain_imgs/69032-2"
    csv_dir_root = os.path.join(root_dir, "5 - Data")
    segment_bead_csv(root_dir)
    #
    # csv_dir_root = os.path.join(root_dir, "5 - Data")
    # for item in os.listdir(csv_dir_root):
    #     if 'Location' in item:
    #         csv_path_1 = os.path.join(csv_dir_root, item)
    #         break
    #
    # csv_path_2 = "test.csv"
    # comparision("/mnt/4T/brain_imgs/73594-2/5 - Data/73594-2 - Manual Bead Location Data v0.1.4 - Dilation Factor 2.csv", "/mnt/4T/brain_imgs/73594-2/5 - Data/auto.csv", img_path_list)
