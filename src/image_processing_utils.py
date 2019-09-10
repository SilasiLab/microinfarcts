'''
This script contains the necessary image processing functions in both preprocessing stage and post processing stage.
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab
'''
# The pixel poistion of the original point in allen atlas
ATLAS_CERTER_POSITION = (208, 212, 228)
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
import pickle as pk
from tqdm import tqdm
import skimage.io as io
from present import show_img, merge_layers, load_img
from bead_finder import save_bead_mask
from ants_utils import quick, apply_transform

def is_in_center(centerPonit, real_img):
    '''
    Given a center point and the image containing the brain, determine whether the region is in the cernter of image or not
    :param centerPonit:
    :param real_img:
    :return: boolean
    '''
    x = float(real_img.shape[1] * 0.5)
    y = float(real_img.shape[0] * 0.5)
    if (centerPonit[0] > (x * 0.6)) and (centerPonit[0] < (x * 1.4)) and (centerPonit[1] > (y * 0.4)) and (centerPonit[1] < (y * 1.4)):
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
                if show:
                    cv2.rectangle(tissue_frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
                    show_img(tissue_frame, False)

    cur_h = 0
    cur_w = 0
    (height, length) = atlas_frame.shape
    for row in range(height):
        if np.asarray(atlas_frame[row, :]).sum() > 0:
            cur_h += 1
    for col in range(length):
        if np.asarray(atlas_frame[:, col]).sum() > 0:
            cur_w += 1
    try:
        w_factor = float(w)/float(cur_w)
        h_factor = float(h) / float(cur_h)
    except:
        tissue_frame = img_frame.copy()

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(tissue_frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
            show_img(tissue_frame, False)

    refactored_atlas_center = (int(ATLAS_CERTER_POSITION[2] * w_factor), int(ATLAS_CERTER_POSITION[1] * h_factor))

    atlas_frame = cv2.resize(atlas_frame, (int(atlas_frame.shape[1] * w_factor), int(atlas_frame.shape[0] * h_factor)), interpolation=cv2.INTER_NEAREST)
    ann_frame = cv2.resize(ann_frame, (int(ann_frame.shape[1] * w_factor), int(ann_frame.shape[0] * h_factor)), interpolation=cv2.INTER_NEAREST)

    atlas_frame, atlas_center = get_pure_brain_atlas(atlas_frame, refactored_atlas_center)
    ann_frame, _ = get_pure_brain_atlas(ann_frame, refactored_atlas_center)

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
         show_atlas=False, intro=True):
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

        save_bead_mask(save_directory, os.path.join(root_dir, name), show_circle=show)


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

if __name__ == '__main__':
    pass

