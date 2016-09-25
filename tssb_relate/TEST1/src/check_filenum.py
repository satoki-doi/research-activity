# !/usr/bin/env python
# coding: UTF-8

import os
import os.path
import pandas as pd

def CheckFileNum(dir):
    InDir_FileList = os.listdir(dir)
    count = 0
    miss_files = 0
    miss = []
    for dir_name in InDir_FileList:
        Imgpath = dir + dir_name + '/'
        if os.path.isdir(Imgpath) and len(os.listdir(Imgpath)) < 10:
            print Imgpath, len(os.listdir(Imgpath))
            count += 1
            miss_files += 10 - len(os.listdir(Imgpath))
            if len(os.listdir(Imgpath)) < 3:
                miss.append(dir_name)
    print count, "miss img is %s" % miss_files
    return miss

if __name__ == '__main__':
    dir = "./crawl_data/"
    miss = CheckFileNum(dir)
    miss = pd.DataFrame(miss)
    print miss.shape
    miss.to_csv("./input_data/miss.csv", index=False)