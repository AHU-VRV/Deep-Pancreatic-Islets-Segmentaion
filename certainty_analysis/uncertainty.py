#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import skimage.io as io
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

probs_dir = '/home/qing/code/Project_AANet/predit_score_map/AAM/ProbMaps/' # the predicted class probability map of each image, the dimension is (H, W, 2)
label_dir = '/home/qing/code/Project_AANet/data/test/label/'    # label of test image

files = os.listdir(probs_dir)
for i, file in enumerate(files):

    index = file.rfind('.')
    name_only = file[:index]
    print(name_only)

    label = cv2.imread(os.path.join(label_dir,name_only+'.png'),cv2.IMREAD_GRAYSCALE)
    num_islet = np.sum(label>0)

    fname = os.path.join(probs_dir, file)

    # Load
    probs = np.load(fname)
    probs_islet = probs[:,:,1]


    #
    max_probs = np.max(probs_islet)
    conf = abs(probs_islet-max_probs) / max_probs

    num_certain_islet = np.sum(conf[label>0]<0.5)

    uncertain = round(num_certain_islet/num_islet,2)
    with open('/home/qing/code/CENet_pancreas/predit_score_map/AAM/certain.txt', 'a+') as f:
        f.write('%s: %s'%(name_only,str(uncertain))+ '\n')

    del probs