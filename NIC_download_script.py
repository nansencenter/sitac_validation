#!/usr/bin/env python
# coding: utf-8

import urllib.request
import zipfile
import os
import argparse
from datetime import datetime, timedelta
import time
import random

def download_usnic_data(date, path):
    year = date.year
    month = date.month
    day = date.day
    url = f'https://usicecenter.gov/File/DownloadArchive?prd=26{month:02}{day:02}{year:04}'
    ofile = f'{path}/arctic{str(year)[2:]}{month:02}{day:02}.zip'
    print(url, ofile)
    
    # Download file
    try:
        urllib.request.urlretrieve(url, ofile)
        print(f'File {ofile} downloaded')
    except:
        print('Error 404')
        print('No data available at this date')
        time.sleep(random.uniform(0,0.5))
        return
    
    # Create a folder
    directory = f'arctic{str(year)[2:]}{month:02}{day:02}'
    path_file = os.path.join(path, directory)
    os.makedirs(path_file, exist_ok=True)
    print("Directory '% s' created" % directory)    
    
    # Unzip in this folder
    with zipfile.ZipFile(ofile, 'r') as zip_ref:
        zip_ref.extractall(path + directory)
    print('File unzip')
    
    # Remove zip file
    #os.remove(path + 'arctic_zip')

parser = argparse.ArgumentParser()
parser.add_argument("start_date", help="startd date of download yyyy-mm-dd")
parser.add_argument("end_date", help="end date of download yyyy-mm-dd")
parser.add_argument("path", help="path to store data /path/")
args = parser.parse_args()

start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
date = start_date
while date <= end_date:
    download_usnic_data(date, args.path)
    date += timedelta(1)