'''
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab
'''
# from nipype.interfaces.ants import RegistrationSynQuick, Registration
# from nipype.interfaces.ants.resampling import ApplyTransforms, ApplyTransformsToPoints
import os
# import shlex
# import subprocess
import ants
import numpy as np
import cv2
from tifffile import imsave, imread

# def quick(dir_fix_image='/home/silasi/ants_data/nrrd/0.tif', dir_moving_img='/home/silasi/ants_data/tissue/0.tif', dir_output='/home/silasi/ants_data/output_', ANTs_script="/home/silasi/ANTs/Scripts/"):
#     """
#     Ants registration function
#     :param dir_fix_image:
#     :param dir_moving_img:
#     :param dir_output:
#     :return:
#     """
#     reg = RegistrationSynQuick()
#     reg.inputs.dimension = 2
#     reg.inputs.fixed_image = dir_fix_image
#     reg.inputs.moving_image = dir_moving_img
#     reg.inputs.output_prefix = dir_output
#     reg.inputs.transform_type = 's'
#     reg.inputs.num_threads = 16
#     command = os.path.join(ANTs_script, reg.cmdline)
#     args = shlex.split(command)
#     print(command)
#     p = subprocess.Popen(args)
#     p.wait()

# def apply_transform(input_img, reference_img, transforms, output_img):
#     """

#     :param input_img:
#     :param reference_img:
#     :param transforms: should be a list of .mat and warp.nii.gz
#     :param output_img:
#     :return:
#     """
#     at1 = ApplyTransforms()
#     at1.inputs.dimension = 2
#     at1.inputs.input_image = input_img
#     at1.inputs.reference_image = reference_img
#     at1.inputs.output_image = output_img
#     # at1.inputs.interpolation = 'BSpline'
#     # at1.inputs.interpolation_parameters = (5,)
#     at1.inputs.default_value = 0
#     at1.inputs.transforms = transforms
#     at1.inputs.invert_transform_flags = [False, False]
#     args = shlex.split(at1.cmdline)
#     p = subprocess.Popen(args)
#     p.wait()

# def apply_transform_2_points(input_csv, transforms, output_csv):

#     at = ApplyTransformsToPoints()
#     at.inputs.dimension = 2
#     at.inputs.input_file = input_csv
#     at.inputs.transforms = transforms
#     at.inputs.invert_transform_flags = [False, False]
#     at.inputs.output_file = output_csv
#     args = shlex.split(at.cmdline)
#     p = subprocess.Popen(args)
#     p.wait()

# def slow(dir_fix_image='/home/silasi/ants_data/nrrd/0.tif', dir_moving_img='/home/silasi/ants_data/tissue/0.tif', dir_output='/home/silasi/ants_data/output_', ANTs_script="/home/silasi/ANTs/Scripts/"):
#     """
#     Ants registration function
#     :param dir_fix_image:
#     :param dir_moving_img:
#     :param dir_output:
#     :return:
#     """
#     reg = Registration()
#     reg.inputs.dimension = 2
#     reg.inputs.fixed_image = dir_fix_image
#     reg.inputs.moving_image = dir_moving_img
#     reg.inputs.output_transform_prefix = dir_output
#     reg.inputs.transforms = ['Affine', 'SyN']
#     reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
#     reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
#     reg.inputs.metric = ['Mattes']*2
#     reg.inputs.metric_weight = [1]*2
#     reg.inputs.radius_or_number_of_bins = [32]*2
#     reg.inputs.sampling_strategy = ['Random', None]
#     reg.inputs.sampling_percentage = [0.05, None]
#     reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
#     reg.inputs.convergence_window_size = [20]*2
#     reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
#     reg.inputs.sigma_units = ['vox'] * 2
#     reg.inputs.shrink_factors = [[2,1], [3,2,1]]
#     reg.inputs.use_estimate_learning_rate_once = [True, True]
#     reg.inputs.use_histogram_matching = [True, True]

#     reg.inputs.num_threads = 16
#     command = os.path.join(ANTs_script, reg.cmdline)
#     args = shlex.split(command)
#     print(command)
#     p = subprocess.Popen(args)
#     p.wait()

def pyAntsReg(fixed_image, moving_image, output_prefix):
    fixed_image = imread(fixed_image)
    moving_image = imread(moving_image)
    fixed_image = ants.from_numpy(fixed_image)
    moving_image = ants.from_numpy(moving_image)
    # moving_image = ants.image_read(moving_image)
    # fixed_image = ants.image_read(fixed_image)
    # fixed_image.plot(overlay=moving_image, title='Before Registration')
    # cv2.waitKey()

    reg = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')
    warped_moving_image = reg['warpedmovout']
    
    ants.image_write(warped_moving_image, output_prefix+'.npy')
    warped_moving_image_numpy = warped_moving_image.numpy()
    warped_moving_image_tif = output_prefix + '.tif'
    imsave(warped_moving_image_tif, warped_moving_image_numpy)

    
    # fixed_image.plot(overlay=warped_moving_image, title='After Registration')
    # cv2.waitKey()
    return reg

def pyAntsApp(reg, fixed_image, moving_image, output_filename):
    # moving_image = ants.image_read(moving_image)
    # fixed_image = ants.image_read(fixed_image)
    fixed_image = imread(fixed_image)
    moving_image = imread(moving_image)
    fixed_image = ants.from_numpy(fixed_image)
    moving_image = ants.from_numpy(moving_image)

    warped_moving_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=reg['fwdtransforms'])
    # warped_moving_image.plot()
    # cv2.waitKey()
    assert output_filename.endswith('.npy')
    ants.image_write(warped_moving_image, output_filename)
    warped_moving_image_numpy = warped_moving_image.numpy()
    imsave(output_filename.replace('.npy', '.tif'), warped_moving_image_numpy)