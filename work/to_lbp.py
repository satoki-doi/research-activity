# -*- coding: utf-8 -*-

####
# 2019/11/23
# Satoki Doi
# local binary patterns
####

import cv2
from skimage.feature import local_binary_pattern


def img_to_lbp(img_path, radius, no_points):
    img = cv2.imread(img_path)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    return lbp
