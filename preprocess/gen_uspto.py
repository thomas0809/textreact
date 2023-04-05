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
BASE_MOL = '/scratch/yujieq/uspto_mol/'


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
    for year in range(1976, 2002):
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
    for year in range(2009, 2010):
        path = os.path.join(BASE, str(year))
        for datefile in sorted(os.listdir(path)):
            if is_zip(datefile):
                if datefile < 'I20080930':
                    continue
                with zipfile.ZipFile(os.path.join(path, datefile), 'r') as zipobj:
                    zipobj.extractall(path)
                date = datefile[:-4]
                molpath = os.path.join(BASE_MOL, str(year), date)
                cnt = 0
                total = 0
                for file in glob.glob(f"{path}/project/pdds/ICEwithdraw/{date}/**/US*.ZIP"):
                    total += 1
                    with zipfile.ZipFile(file, 'r') as zipobj:
                        filelist = zipobj.namelist()
                        if any([name[-4:] in ['.mol', '.MOL'] for name in filelist]):
                            zipobj.extractall(molpath)
                            cnt += 1
                print(datefile, f"{cnt} / {total} have molecules")


if __name__ == "__main__":
    if sys.argv[1] == 'download':
        download()
    elif sys.argv[1] == 'download_fulltext':
        download_fulltext()
    elif sys.argv[1] == 'unzip':
        unzip()
