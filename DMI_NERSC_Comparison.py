#!/usr/bin/env python
# coding: utf-8

# # Import

# In[2]:


import glob
import argparse
from datetime import date, datetime, timedelta
from cartopy.crs import NorthPolarStereo, LambertAzimuthalEqualArea
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.image
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sklearn.metrics as skm
#from PIL import Image
from matplotlib.gridspec import GridSpec


# # Functions

# In[5]:


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def nc_ice_comparison(start_date, end_date, path_man, path_aut, path_stats):
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y-%m-%d"))
        day = single_date.strftime("%d")
        month = single_date.strftime("%m")
        year = single_date.strftime("%Y")
        
        path_man_files = path_man + year + '/' + month + '/ice_conc_greenland_' + year + month + day + '*.nc'
        path_aut_files = path_aut + year + '/' + month + '/s1_icetype_mosaic_'+ year + month + day + '0600.nc'
    
        aut_files = sorted(glob.glob(path_aut_files))
        man_files = sorted(glob.glob(path_man_files))

        if len(man_files) == 0:
            print('SKIP')
        else :
            daily_ice_comparison(day, month, year, path_man, path_aut, path_stats)
            
def daily_ice_comparison(day, month, year, path_man, path_aut, path_stats):
    
    path_man = path_man + year + '/' + month + '/ice_conc_greenland_' + year + month + day + '*.nc'
    path_aut = path_aut + year + '/' + month + '/s1_icetype_mosaic_'+ year + month + day + '0600.nc'
    
    ifiles_manu = sorted(glob.glob(path_man))
    with Dataset(ifiles_manu[0]) as ds_man:
        ds_man = Dataset(ifiles_manu[0])
        x_man = ds_man['xc'][:]
        y_man = ds_man['yc'][:]
        grid_size = ds_man['ice_poly_id_grid'][0].shape
        
    with Dataset(path_aut) as ds_auto:
        x_aut = ds_auto['xc'][:]
        y_aut = ds_auto['yc'][:]
        mask_aut = ds_auto['confidence'][0].filled(0) > 0
        ice_type = ds_auto['ice_type'][0].filled(0)
        confidence = ds_auto['confidence'][0].filled(0)

    print('making mosaic')
    mosaic, mask_mosaic = make_mosaic (ifiles_manu, grid_size)
    
    print('reprojecting')
    mosaic_inter, mask_mosaic_inter = reproject(mosaic, mask_mosaic, y_man, x_man, x_aut, y_aut)
    
    print('ice difference')
    man2aut, res_man, res_aut, mask_diff, mask_nan, land_mask = ice_difference (mosaic_inter, ice_type, mask_mosaic_inter, mask_aut)

    intersec = np.count_nonzero(mask_diff == 1)
    if intersec == 0 :
        print('DATE ' + year + month + day + ' SKIPPED' )
    else :
        print('Statistics') 
        data = compute_stats_all (man2aut, res_man, res_aut, mask_diff, confidence)
        
        print('writing data')
        write_stats_day(data, path_stats + 'stats_m_', year + month + day)
        
        print('saving images')
        image_render(year, month, day, path_stats, man2aut, res_man, res_aut, land_mask, mask_diff)

def get_man_file(path):
    
    with Dataset(path) as ds:

        ct = ds['CT'][0]
        ca = ds['CA'][0]
        sa = ds['SA'][0]
        cb = ds['CB'][0]
        sb = ds['SB'][0]
        cc = ds['CC'][0]
        sc = ds['SC'][0]
        polygon_id = ds['polygon_id'][0]
        polygon_reference = ds['polygon_reference'][:]
        ice_poly_id_grid = ds ['ice_poly_id_grid'][0]
        
    return ct,ca,sa,cb,sb,cc,sc,polygon_id, polygon_reference, ice_poly_id_grid



def si_type(stage):
    
    index_ = 0
    
    if stage ==0:
        index_ = 0
    #print('ice_free')

    if stage in range(81,86):
        #print('Young ice')
        index_=1
    if stage in range(86,94):
        #print('First year ice')
        index_=2
    if stage in range(95,98):
        #print('multiyear ice')
        index_=3
    return index_



def dominant_ice(ct,ca,sa,cb,sb,cc,sc,polygon_id, polygon_reference, ice_poly_id_grid):

    dominant_grid = np.zeros(ice_poly_id_grid.shape)
    dominant_vector = np.zeros((len(ca))).astype('int')
    sod = [sa, sb, sc]
    
    for i in range (len(ca)) :
        #dominant_vector[i] = np.argmax([ca[i], cb[i], cc[i]])
        ice = np.argmax([ca[i], cb[i], cc[i]])
        ice_type = si_type(sod[ice][i])
        dominant_vector[i] = ice_type
    
    for p_ref in polygon_reference:
        # ic take indices
        ic = np.where(p_ref == polygon_reference)[0]
        p_id = polygon_id[ic]
        if p_id == -9:
            continue
        mask = ice_poly_id_grid == p_id
        dominant_grid[mask] = dominant_vector[ic]
        
    return dominant_grid



def make_mosaic (files, grid_size):
    
    # Create empty array with the size of a grid
    mosaic = np.zeros(grid_size)
    mask_mosaic = np.zeros(grid_size)
    #mask_mosaic[:] = -1

    # for each file in files list of the day
    for file in files:
        ct,ca,sa,cb,sb,cc,sc,polygon_id, polygon_reference, ice_poly_id_grid = get_man_file(file)
        file_dominant = dominant_ice(ct,ca,sa,cb,sb,cc,sc,polygon_id, polygon_reference, ice_poly_id_grid)
        """mask = file_dominant >= 1 
        may come from a previous version of the mask"""
        
        mask = ice_poly_id_grid.filled(-1) >= 0
        mosaic[mask] = file_dominant[mask]

        mask_mosaic[mask] = 1
        #mask_mosaic = mask_mosaic + file_grid[0]
        
    return mosaic, mask_mosaic



def reproject(mosaic, mask_mosaic, y_man, x_man, x_aut, y_aut):
    
    # Define projection of the sea ice drift product +proj=stere +lat_0=90n +lon_0=0e +lat_ts=90n +r=6371000
    crs_aut = NorthPolarStereo(0, 90)
    # define projection of the thickness product +proj=stere +lon_0=-45 +lat_ts=90 +lat_0=90 +a=6371000 +b=6371000
    crs_man = NorthPolarStereo(-45, 90)
    
    # create matrices of coordinates for reprojection of SIT product from LAEA to NPS projection
    # NPS coordinates on NPS grid
    x_aut_grd, y_aut_grd = np.meshgrid(x_aut, y_aut)
    # LAEA coordinates on NPS grid
    grd_man = crs_man.transform_points(crs_aut, x_aut_grd, y_aut_grd)
    x_grd_man, y_grd_man = grd_man[:,:,0], grd_man[:,:,1]
    
    # Prepare interpolators for thickness and concentration
    rgi = RegularGridInterpolator((y_man, x_man), mosaic, method='nearest', bounds_error=False)
    mask_rgi = RegularGridInterpolator((y_man, x_man), mask_mosaic, method='nearest', bounds_error=False)
    # Do interpolation from LAEA grid onto NPS grid
    mosaic_inter = rgi((y_grd_man, x_grd_man))
    mask_mosaic_inter = mask_rgi((y_grd_man, x_grd_man))
    
    return mosaic_inter, mask_mosaic_inter



def ice_difference (mosaic_inter, ice_type, mask_mosaic_inter, mask_aut):
    
    # Create different masks
    mask_man = mask_mosaic_inter > 0
    mask_common = mask_man * mask_aut
    mask_nan = np.where(mask_common == 0, np.nan, mask_common)
    land_mask = ice_type == -1
    
    # Results 
    res_man = mosaic_inter * mask_common
    res_man = np.nan_to_num(res_man, nan=0)
    res_aut = ice_type * mask_common

    diff_man_aut = res_man - res_aut
    
    return diff_man_aut, res_man, res_aut, mask_common, mask_nan, land_mask



def compute_stats_all (man2aut, res_man, res_aut, mask_diff, confidence):
    
    m_man = res_man[mask_diff]
    m_aut = res_aut[mask_diff]
    m_conf = confidence[mask_diff]
    
    # basic metric  
    report = skm.classification_report(m_man, m_aut, digits=3, output_dict=True)
    
    accuracy = report['accuracy']
    macro_avg_p = report['macro avg']['precision']
    macro_avg_r = report['macro avg']['recall']
    macro_avg_f = report['macro avg']['f1-score']
    
    weighted_avg_p = report['weighted avg']['precision']
    weighted_avg_r = report['weighted avg']['recall']
    weighted_avg_f = report['weighted avg']['f1-score']
    
    # confusion matrix   
    matrix = skm.confusion_matrix(m_man, m_aut)
    
    # jaccard    possibly ok
    jaccard_labels = skm.jaccard_score(m_man, m_aut, average=None)   # list
    jaccard_avg = skm.jaccard_score(m_man, m_aut, average='weighted')  #float
    
    # Kappa    ok
    kappa = skm.cohen_kappa_score(m_man, m_aut, labels=None, weights=None, sample_weight=None)
    
    # Precision recall fscore   ok                 list
    p, r, f, s = skm.precision_recall_fscore_support(m_man, m_aut, average=None, warn_for=('precision', 'recall', 'f-score'))

    # matthews_corrcoef    ok
    mcc = skm.matthews_corrcoef(m_man, m_aut)
    
    # hamming_loss    possibly ok
    hloss = skm.hamming_loss(m_man, m_aut)
    
    # balanced accuracy
    b_acc = skm.balanced_accuracy_score(m_man, m_aut)
    
    log_loss_binary, log_loss_percentage, auc_roc_binary, auc_roc_percentage = confidence_metrics(m_man, m_aut, m_conf)
    
    # Count px in comparison, manual and automatic images

    total_man = []
    total_aut = []
    total = []
    ind = []

    for i in range (-3,4):
        count = np.count_nonzero(man2aut[mask_diff] == i)
        total.append(count)
        ind.append(i)

    for i in range (4):
        count_man = np.count_nonzero(res_man[mask_diff] == i)
        count_aut = np.count_nonzero(res_aut[mask_diff] == i)
        total_man.append(count_man)
        total_aut.append(count_aut)
    
    report_avg = [accuracy, macro_avg_p, macro_avg_r, macro_avg_f, weighted_avg_p, weighted_avg_r, weighted_avg_f]
    conf_indexes = [log_loss_binary, log_loss_percentage, auc_roc_binary, auc_roc_percentage]
    mat_lis = [p, r, f, s, matrix, jaccard_labels, total, total_man, total_aut] 
    indexes = [b_acc, hloss, mcc, kappa, jaccard_avg]
    
    datas = report_avg + conf_indexes + mat_lis + indexes
    
    return datas



def confidence_metrics(m_man, m_aut, m_conf):
    
    binary = []
    percentage = []

    for i in range (len(m_man)):
        proba = [0,0,0,0]
        proba[m_aut[i]] = 1
        binary.append(proba)

        max_conf = m_conf[i]*0.01
        min_conf = (1-max_conf)/3

        proba_ = [min_conf,min_conf,min_conf,min_conf]
        proba_[m_aut[i]] = max_conf
        percentage.append(proba_)

    log_loss_binary = skm.log_loss(m_man, binary,  labels=np.array([0. ,1. ,2. ,3.]))
    log_loss_percentage = skm.log_loss(m_man, percentage, labels=np.array([0. ,1. ,2. ,3.]))

    try :
        auc_roc_binary = skm.roc_auc_score(m_man, binary, multi_class='ovr')
    except ValueError:
        auc_roc_binary = 0.0
    try:
        auc_roc_percentage = skm.roc_auc_score(m_man, percentage, multi_class='ovr')
    except ValueError:
        auc_roc_percentage = 0.0
    
    return log_loss_binary, log_loss_percentage, auc_roc_binary, auc_roc_percentage



def write_stats_day(datas, path_stats, filename):
    
    with open(path_stats + filename + '.txt', "a") as file:
        
        for data in datas:
    
            #print(type(data))
    
            if type(data) == float or isinstance(data, np.float64):
                file.write(str(data) + "\n")

            if type(data) == list:
                count = ';'.join(map(str, data))
                file.write(count+"\n")

            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    count = ';'.join(map(str, data))
                    file.write(count+"\n")

            if isinstance(data, np.ndarray):
                if data.ndim == 2:
                    file.write("Confusion matrix\n")
                    rows = ["{};{};{};{}".format(i, j, k, l) for i, j, k, l in data]
                    conf = "\n".join(rows)
                    file.write(conf)
                    file.write("\n")



def image_render(year, month, day, path_img, man2aut, res_man, res_aut, land_mask, mask_diff):

    # adapt array with values for no data and land

    # difference between manual and automatic
    img = man2aut
    img[~mask_diff] = -4
    img[land_mask] = -5

    # manual
    img_man = res_man
    img_man[~mask_diff] = -1
    img_man[land_mask] = -2

    # automatic
    img_aut = res_aut
    img_aut[~mask_diff] = -1
    img_aut[land_mask] = -2


    # Colormap for comparison
    cmap = plt.cm.colors.ListedColormap(['gray','white' ,
                                         '#b30727', '#e8775d', '#f0cab7', '#cfd9e8', '#b5cdf8', '#6485ec', '#384abf'])
    # Colormap for ice type (from H.Boulze)
    cmap_hugo = plt.cm.colors.ListedColormap(['whitesmoke', 'white', '#0064ff', '#aa28f0', '#ffff00', '#b46432'])

    # Normalization of ice comparison
    norm = plt.Normalize(-5, 4)
    img_norm = norm(img)
    img_ = cmap(img_norm)

    # Normalization of ice types
    norm2 = plt.Normalize(-2,4)
    img_man_norm = norm2(img_man)
    img_man = cmap_hugo(img_man_norm)
    img_aut_norm = norm2(img_aut)
    img_aut = cmap_hugo(img_aut_norm)

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, width_ratios=[4, 3, 3], height_ratios=[3, 3, 3])
    fig.suptitle("Ice comparison " + year + "-" + month + "-" + day, fontsize='x-large')

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(img_, cmap=cmap, aspect='auto')
    ax1.set_title('Comparison')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(img_man, cmap=cmap_hugo, aspect='auto')
    ax2.set_title('Manual classification')

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(img_aut, cmap=cmap_hugo, aspect='auto')
    ax3.set_title('Automatic classification')

    cbar_comp = plt.colorbar(im1, ax=ax1)

    cbar_comp.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['ground','no data','-3','-2', '-1', '0', '1', '2', '3']):
        cbar_comp.ax.text(1.3, (j + 0.5) / 9.0, lab, ha='left', va='center', fontsize='small')

    cbaxes = fig.add_axes([0.5, 0.62, 0.4, 0.02])
    cbar = plt.colorbar(im2, ax=[ax2, ax3], orientation='horizontal', cax = cbaxes)

    cbar.ax.get_xaxis().set_ticks([])
    for j, lab in enumerate(['Ground','No Data','Ice free','Young Ice', 'First Year Ice', 'Multi Year Ice']):
        cbar.ax.text((j + 0.5) / 6.0, .5, lab, ha='center', va='center', fontsize='small')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    plt.savefig(path_img+"map_"+year+month+day+".png", dpi=300, bbox_inches='tight')


# # Run

# In[6]:


#day = '30'
#month = '01'
#start_date = date(2023, 1, 1)
#end_date = date(2023, 2, 5)

#path_aut = '/Data/sat/auxdata/ice_charts/NERSC/nrt.cmems-du.eu/Core/SEAICE_ARC_PHY_AUTO_L4_NRT_011_015/cmems_obs-si_arc_phy-icetype_nrt_L4-auto_P1D/'
#path_man = '/Data/sat/auxdata/ice_charts/DMI/nrt.cmems-du.eu/Core/SEAICE_ARC_SEAICE_L4_NRT_OBSERVATIONS_011_002/cmems_obs-si_arc_physic_nrt_1km-grl_P1D-irr/'
#path_stats = '/home/malela/data/comp2/'
#path_stats = '/home/malela/data/img/'


# In[13]:


#nc_ice_comparison(start_date, end_date, path_man, path_aut, path_stats)

parser = argparse.ArgumentParser()
parser.add_argument("start", help="start date of computation YYYY-mm-dd")
parser.add_argument("end", help="end date of computation YYYY-mm-dd")
parser.add_argument("path_man", help="path of manuals data /path/to/data/")
parser.add_argument("path_aut", help="path of automatics data /path/to/data/")
parser.add_argument("path_stats", help="path of statistics results and images /path/to/data/")
args = parser.parse_args()

istart = args.start
istart = istart.split('-')
start_date = date(int(istart[0]), int(istart[1]), int(istart[2]))
iend = args.end
iend = iend.split('-')
end_date = date(int(iend[0]), int(iend[1]), int(iend[2]))

nc_ice_comparison(start_date, end_date, args.path_man, args.path_aut, args.path_stats)


# In[ ]:




