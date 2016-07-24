#!/usr/bin/env python
# coding: UTF-8

import os

def CheckFileNum(dir):
    InDir_FileList = os.listdir(dir)
    for File in InDir_FileList:
        Imgpath = dir + File
        FileNum = os.listdir(Imgpath)
        if FileNum < 20:
            print Imgpath

if __name__ == '__main__':
    dir = "../output_data/"
    CheckFileNum(dir)