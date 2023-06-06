import urllib.request
import zipfile
import os
import re
import sys
import glob
import shutil
import multiprocessing
import pandas as pd
from tqdm import tqdm
from collections import Counter
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import numpy as np


BASE = '/scratch/yujieq/uspto_grant_red/'
BASE_TXT = '/scratch/yujieq/uspto_grant_fulltext/'
BASE_TXT_ZIP = '/scratch/yujieq/uspto_grant_fulltext_zip/'


# Download data
def _download_file(url, output):
    if not os.path.exists(output):
        urllib.request.urlretrieve(url, output)


def download():
    for year in range(2016, 2017):
        url = f"https://bulkdata.uspto.gov/data/patent/grant/redbook/{year}/"
        f = urllib.request.urlopen(url)
        content = f.read().decode('utf-8')
        print(url)
        zip_files = re.findall(r"href=\"(I*\d\d\d\d\d\d\d\d(.ZIP|.zip|.tar))\"", content)
        print(zip_files)
        path = os.path.join(BASE, str(year))
        os.makedirs(path, exist_ok=True)
        args = []
        for file, ext in zip_files:
            output = os.path.join(path, file)
            args.append((url + file, output))
        # with multiprocessing.Pool(8) as p:
        #     p.starmap(_download_file, args)
        for url, output in args:
            print(url)
            _download_file(url, output)


def download_fulltext():
    for year in range(2002, 2017):
        url = f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/'
        f = urllib.request.urlopen(url)
        content = f.read().decode('utf-8')
        print(url)
        zip_files = re.findall(r"href=\"(\w*(.ZIP|.zip|.tar))\"", content)
        print(zip_files)
        path = os.path.join(BASE_TXT, str(year))
        os.makedirs(path, exist_ok=True)
        args = []
        for file, ext in zip_files:
            output = os.path.join(path, file)
            args.append((url + file, output))
        # with multiprocessing.Pool(8) as p:
        #     p.starmap(_download_file, args)
        for url, output in args:
            print(url)
            _download_file(url, output)


# Unzip
def is_zip(file):
    return file[-4:] in ['.zip', '.ZIP']


def unzip():
    for year in range(1976, 2017):
        path = os.path.join(BASE_TXT_ZIP, str(year))
        outpath = os.path.join(BASE_TXT, str(year))
        for datefile in sorted(os.listdir(path)):
            if is_zip(datefile):
                print(os.path.join(path, datefile))
                with zipfile.ZipFile(os.path.join(path, datefile), 'r') as zipobj:
                    zipobj.extractall(outpath)


if __name__ == "__main__":
    if sys.argv[1] == 'download':
        download()
    elif sys.argv[1] == 'download_fulltext':
        download_fulltext()
    elif sys.argv[1] == 'unzip':
        unzip()
