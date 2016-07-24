#!/usr/bin/env python
# coding: UTF-8

import time
import urllib
import urllib2
import sys
from bs4 import BeautifulSoup
import os.path
import cPickle
from numpy.random import *
import ssl
from functools import wraps

dir = "../output_data"
OPENER = urllib2.build_opener()
OPENER.addheaders = [("User-Agent", "Mozilla/5.0")]
base = 'https://www.google.co.jp/search?site=imghp&tbm=isch&source=hp&num=30&q='


def SEARCH_WORD_GetHTML(que):
    """
    Google検索にアクセスする
    """
    url = base + urllib.quote("\"%s\"" % que)
    html = OPENER.open( url ).read()
    soup = BeautifulSoup(html, "lxml")
    return soup


def DL_img(img_url, word):
    """
    IMGファイルのダウンロード    
    """
    try:
        img = urllib2.urlopen(img_url)
        fname = img_url.split("/")[-1]
        file = '%s/%s/%s.jpg' % (dir, word, fname)
        if os.path.isfile(file) == False:
            localfile = open(file, "w")
            print file
            localfile.write(img.read())
            img.close()
            localfile.close()
        else:
            print "%s is already exist" % file
    except:
        print "Not Download %s" % img_url

def Get_imgURL(soup, word):
    Namelist = soup.find_all('img')
    for link in Namelist:
        try:
            img_url = link.get('src')
            DL_img(img_url, word)
            stop_time = poisson(lam=1) + 1
            time.sleep(stop_time)
        except Exception as e:
            print e

# 商品名の読み取り
def pickle_load(file_name):
    """
    file loading
    """
    fh = open(file_name)
    result = cPickle.load(fh)
    fh.close
    return result

def sslwrap(func):
    @wraps(func)
    def bar(*args, **kw):
        kw['ssl_version'] = ssl.PROTOCOL_TLSv1
        return func(*args, **kw)
    return bar

def main(item_list):
    """
    web scraiping start
    """
    count = 0
    for item in item_list:
        print print_text % item + '[%s]' % count
        item_dir = '../output_data/%s' % item
        if os.path.exists(item_dir) == False:
            os.mkdir(item_dir)
        else:
            print "dir already exist"
        soup = SEARCH_WORD_GetHTML(item)
        Get_imgURL(soup, item)
        stop_time = poisson(lam=5) + 1
        time.sleep(stop_time)
        count += 1    

if __name__ == '__main__':
    file_name = "../input_data/item_list.pkl"
    item_data = pickle_load(file_name)
    item_list = item_data['item_name'].values()
    print_text = """
        -----------------------------------
        -----------------------------------
        scraipin %s ....... 
        -----------------------------------
        -----------------------------------
        """
    ssl.wrap_socket = sslwrap(ssl.wrap_socket)
    main(item_list)
