# -*- coding: utf-8 -*-

import os
import glob
import random

import numpy as np
import pandas as pd
import cv2
import shutil

PATCH_SIZE = 300  
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE, 3)

if __name__ == '__main__':
    bbox_coords = pd.read_csv('data/coords.csv')
    train_ids = list(set(bbox_coords.image_id.values))

    output = {}
    output_dir = 'cache/patches_{}'.format(PATCH_SIZE)
    os.makedirs(output_dir)
    
    for image_id in train_ids:
        print(image_id)
        img = cv2.imread('data/Train/{}.jpg'.format(image_id))
        
        h, w, _ = img.shape
        
        for j in range(0, h, PATCH_SIZE):
            for i in range(0, w, PATCH_SIZE):
                j2 = j+PATCH_SIZE
                i2 = i+PATCH_SIZE

                if img[j:j2, i:i2, :].shape != PATCH_SHAPE:
                    # ignoring patches on the border of the image for simplicity
                    continue

                # check for dots within patch
                coords = bbox_coords[bbox_coords.image_id == image_id]
                if len(coords.values) == 0:
                    continue

                img_name = '{}_{}_{}.jpg'.format(image_id, j, i)
                img_out = os.path.join(output_dir, img_name)
                rows = []
                for c in coords.values:
                    _, klass_id, x1, y1, x2, y2 = c

                    # dot's x coordinate outside patch
                    if x1 > i2 or x2 < i:
                        continue

                    # dot's y coordinate outside patch
                    if y1 > j2 or y2 < j:
                        continue

                    px1 = (x1 - i)
                    px2 = (x2 - i)
                    py1 = (y1 - j)
                    py2 = (y2 - j)

                    rows.append([klass_id, px1, py1, px2, py2])

                if len(rows) > 0:
                    cv2.imwrite(img_out, img[j:j2, i:i2, :])                    
                    # write labels as well
