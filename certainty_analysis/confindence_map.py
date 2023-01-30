#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

probs_dir = "/home/qing/code/AANet_pancreas/predit_score_map/AAM/ProbMaps/"     # The predicted class probability map of each image, the dimension is (H, W, 2)
confidence_dir = "/home/qing/code/AANet_pancreas/predit_score_map/AAM/Confidence_Maps/"


files = os.listdir(probs_dir)
for i, file in enumerate(files):

    index = file.rfind('.')
    name_only = file[:index]
    print(name_only)

    fname = os.path.join(probs_dir, file)

    # Load
    probs = np.load(fname)
    probs_islet = probs[:,:,1]

    #Subtract the maximum probability value belonging to the foreground from the probability that the pixel belongs to the foreground
    max_probs = np.max(probs_islet)
    conf = abs(probs_islet-max_probs) / max_probs

    # Plot results
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(conf, cmap="magma_r", vmin=0, vmax=1)

    # Modify color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    colbar = fig.colorbar(im, cax=cax)
    colbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    colbar.outline.set_edgecolor((0,0,0,0))
    colbar.ax.tick_params(length=0)

    ax.axis('off')
    plt.show()

    save_name = os.path.join(confidence_dir, file.split('.')[0] + ".png")

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    # plt.savefig(save_name, dpi=300)
    plt.close()

    del probs