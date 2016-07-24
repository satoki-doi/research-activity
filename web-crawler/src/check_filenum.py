#!/usr/bin/env python
# coding: UTF-8

import os
import os.path 

def CheckFileNum(dir):
    InDir_FileList = os.listdir(dir)
    count = 0
    for dir_name in InDir_FileList:
        Imgpath = dir + dir_name + '/'
        if os.path.isdir(Imgpath) and len(os.listdir(Imgpath)) < 20:
            print Imgpath
            count += 1
    print count

if __name__ == '__main__':
    dir = "../output_data/"
    CheckFileNum(dir)