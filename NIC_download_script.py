#!/usr/bin/env python
# coding: utf-8

# In[2]:


import urllib.request
import zipfile
import os
import argparse


# In[3]:


def download_usnic_data(year, month, day, path):  
    url = 'https://usicecenter.gov/File/DownloadArchive?prd=26' + month + day + year

    # Download file
    try :
        urllib.request.urlretrieve(url, path + 'arctic_zip')
        print('File {}-{}-{} downloaded'.format(year, month, day))
    except :
        print('Error 404')
        print('No data available at this date')
        return False
    
    # Create a folder
    directory = 'Arctic_' + year + month + day
    path_file = os.path.join(path, directory)
    os.mkdir(path_file)
    print("Directory '% s' created" % directory)    
    
    # Unzip in this folder
    with zipfile.ZipFile(path + 'arctic_zip', 'r') as zip_ref:
        zip_ref.extractall(path + directory)
    print('File unzip')
    
    # Remove zip file
    os.remove(path + 'arctic_zip')
    return True


# In[4]:


#path = '/home/malela/dl_test/'
#download_usnic_data('2023', '01', '13', path)


# In[11]:


parser = argparse.ArgumentParser()
parser.add_argument("date", help="date of the file yyyy-mm-dd")
parser.add_argument("path", help="path to store data /path/")
args = parser.parse_args()
file_date = args.date
file_date = file_date.split('-')
file_path = args.path
download_usnic_data(file_date[0], file_date[1], file_date[2], file_path)


# In[ ]:




