#!/usr/bin/env python

# Import necessary libraries
import os  # For interacting with the operating system
from datetime import date, timedelta  # For working with dates
import glob  # For searching directories for files
import sklearn.metrics as skm  # For various statistical metrics
from matplotlib.gridspec import GridSpec  # For creating grid layouts in plots
# Importing argparse for handling command line arguments
import argparse  
# Importing numpy for numerical operations
import numpy as np  
# Importing matplotlib for plotting
import matplotlib.pyplot as plt  
# Importing netCDF4 for working with NetCDF files
from netCDF4 import Dataset  
# Importing GDAL, OSR, and OGR from osgeo for geospatial operations
from osgeo import gdal, osr, ogr  
# Set matplotlib backend to 'agg' for non-interactive plotting
import matplotlib
matplotlib.use('agg')

def get_gdal_dataset(x_ul, nx, dx, y_ul, ny, dy, srs_proj4, dtype=gdal.GDT_Float32):
    """
    Get empty gdal dataset with a given extent and projection

    Parameters:
    -----------
    x_ul : float
        x coordinates of upper-left corner of upper-left pixel
        ([0,0] pixel)
    nx   : int
        number of pixels in x-direction (number of columns)
    dx   : float
        step size in x direction (as column index increases)
        (can be negative)
    y_ul : float
        y coordinates of upper-left corner of upper-left pixel
        ([0,0] pixel)
    ny   : int
        number of pixels in y-direction (number of rows)
    dy   : float
        step size in y direction (as row index increases)
        (can be negative)
    srs_proj4 : str
        Projection in Proj4 format
    dtype : GDALDataType, optional
        Data type for the dataset. Default is gdal.GDT_Float32.

    Returns:
    --------
    ds : osgeo.gdal.Dataset
        Empty GDAL dataset with specified extent and projection
    """

    # Create dataset
    dst_ds = gdal.GetDriverByName('MEM').Create('tmp', nx, ny, 1, dtype)

    # Set grid limits
    # For usage of osgeo.gdal.Dataset.SetGeoTransform, see:
    # https://gdal.org/tutorials/geotransforms_tut.html
    dst_ds.SetGeoTransform((x_ul, dx, 0, y_ul, 0, dy))

    # Set projection
    srs = osr.SpatialReference()
    srs.ImportFromProj4(str(srs_proj4))
    srs_wkt = srs.ExportToWkt()
    dst_ds.SetProjection(srs_wkt)

    # Set no_data_value for the band
    band = dst_ds.GetRasterBand(1)
    NoData_value = -999999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    return dst_ds

def rasterize_icehart(shapefile, ds):
    """
    Rasterize the ice chart shapefile and extract attribute values.

    Parameters:
    -----------
    shapefile : str
        Path to the ice chart shapefile.
    ds : osgeo.gdal.Dataset
        GDAL dataset to rasterize the shapefile onto.

    Returns:
    --------
    dst_arr : numpy.ndarray
        Rasterized ice chart.
    field_arr : numpy.ndarray
        Attribute values extracted from the shapefile.
    """
    # Define ice chart attribute names
    field_names = ['CT', 'CA', 'CB', 'CC', 'SA', 'SB', 'SC', 'FA', 'FB', 'FC']
    field_arr = []

    # Open the input shapefile
    ivector = ogr.Open(shapefile, 0)
    ilayer = ivector.GetLayer()

    # Create a temporary memory layer for rasterization
    odriver = ogr.GetDriverByName('MEMORY')
    ovector = odriver.CreateDataSource('memData')
    olayer = ovector.CopyLayer(ilayer, 'burn_ice_layer', ['OVERWRITE=YES'])
    fidef = ogr.FieldDefn('poly_index', ogr.OFTInteger)
    olayer.CreateField(fidef)

    # Iterate over features in the memory layer
    for ft in olayer:
        ft_id = ft.GetFID() + 1
        field_vec = [ft_id]
        # Extract attribute values for each feature
        for field_name in field_names:
            field_val = ft.GetField(field_name)
            if field_val is None:
                field_vec.append(-9)  # Assign a default value if attribute is missing
            else:
                field_vec.append(float(field_val))
        field_arr.append(field_vec)
        ft.SetField('poly_index', ft_id)
        olayer.SetFeature(ft)

    # Rasterize the memory layer onto the GDAL dataset
    gdal.RasterizeLayer(ds, [1], olayer, options=["ATTRIBUTE=poly_index"])
    # Read the rasterized array
    dst_arr = ds.ReadAsArray()

    return dst_arr, np.array(field_arr)


def SI_type(stage):
    """
    Determine the ice type based on stage

    Parameters:
    -----------
    stage : int
        Ice stage value

    Returns:
    --------
    index_ : int
        Ice type index:
        0 - ice_free
        1 - Young ice
        2 - First year ice
        3 - Multiyear ice
    """

    index_ = 0
    
    if stage == 0:
        index_ = 0
    #print('ice_free')

    if 81 <= stage < 86:
        #print('Young ice')
        index_=1
    if 86 <= stage < 94:
        #print('First year ice')
        index_=2
    if 95 <= stage < 98:
        #print('multiyear ice')
        index_=3
    return index_

def ice_type_map(polyindex_arr, icecodes):    
    """
    Map ice type to polygons based on icecodes

    Parameters:
    -----------
    polyindex_arr : numpy.ndarray
        Array containing polygon indices
    icecodes : numpy.ndarray
        Array containing ice codes and stages

    Returns:
    --------
    it_array : numpy.ndarray
        Array containing ice type values for each polygon
    """

    it_array = np.zeros(polyindex_arr.shape, dtype=float)
    it_array[:] = -1

    polyids = np.unique(polyindex_arr)

    for polyid in polyids:
        mask = polyindex_arr == polyid
        i = np.where(icecodes[:, 0] == polyid)[0]
        if len(i) > 0:
            ice = np.argmax([icecodes[i, 2], icecodes[i, 3], icecodes[i, 4]])
            sod = [icecodes[i, 5], icecodes[i, 6], icecodes[i, 7]]
            ice_type = SI_type(sod[ice])

            it_array[mask] = ice_type
    return it_array

def daterange(start_date, end_date):
    """
    Generate a range of dates between start_date (inclusive) and end_date (exclusive).

    Parameters:
    -----------
    start_date : datetime.date
        The start date.
    end_date : datetime.date
        The end date.

    Yields:
    -------
    date : datetime.date
        Dates in the range between start_date and end_date.
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def weekrange(end_date):
    """
    Generate a range of dates for the last week ending at end_date.

    Parameters:
    -----------
    end_date : datetime.date
        The end date.

    Yields:
    -------
    date : datetime.date
        Dates of the last week ending at end_date.
    """
    for n in range(6):
        yield end_date - timedelta(n)


def week_auto_files(str_date, path_aut):
    """
    Get the list of automatic files for the week ending at str_date.

    Parameters:
    -----------
    str_date : str
        Date string in the format 'YYYY-MM-DD'.
    path_aut : str
        Path to the automatic files.

    Returns:
    --------
    aut_files : list
        List of automatic files for the week ending at str_date.
    """
    d = str_date.split('-')
    end_date = date(int(d[0]), int(d[1]), int(d[2]))
    aut_files = []

    for single_date in weekrange(end_date):
        print(single_date.strftime("%Y-%m-%d"))
        day = single_date.strftime("%d")
        month = single_date.strftime("%m")
        year = single_date.strftime("%Y")

        aut_file = sorted(glob.glob(os.path.join(path_aut, 's1_icetype_mosaic_'+year+month+day+'0600.nc'))) 
        if len(aut_file) > 0:
            aut_files.append(aut_file[0])
    return aut_files


def mosaic_auto_argmax(aut_files):
    """
    Create a mosaic of ice types based on automatic files using argmax of probabilities.

    Parameters:
    -----------
    aut_files : list
        List of automatic files.

    Returns:
    --------
    max_prob_idx : numpy.ndarray
        Mosaic of ice types based on the argmax of probabilities.
    """
    # Get the shape of the dataset from the last file
    with Dataset(aut_files[-1]) as ds:
        n, m = ds['ice_type'][0].filled(0).shape

    maps = []
    prob = []

    # Loop through each file
    for file in aut_files:
        with Dataset(file) as ds:
            ice_type = ds['ice_type'][0].filled(4)
            confidence = ds['confidence'][0]
            
            # Set ice type as 4 where confidence is 0 or ice type is -1
            ice_type[confidence == 0] = 4
            ice_type[ice_type == -1] = 4

        maps.append(ice_type)
        prob.append(confidence)
        
    # Create meshgrid for column and row indices
    cols, rows = np.meshgrid(range(m), range(n))
    sum_prob = np.zeros((n, m, 4))

    # Calculate sum of probabilities for each ice type
    for p, m in zip(prob, maps):
        gpi = m < 4
        sum_prob[rows[gpi], cols[gpi], m[gpi]] += p[gpi]
        
    # Get the index of maximum probability for each cell
    max_prob_idx = np.argmax(sum_prob, axis=2)

    # Set index to -1 where sum of probabilities is 0
    max_prob_idx[sum_prob.sum(axis=2) == 0] = -1
    
    return max_prob_idx


def is_difference(mosaic_aut, usice_inter):
    """
    Calculate the difference between two ice type mosaics.

    Parameters:
    -----------
    mosaic_aut : numpy.ndarray
        Ice type mosaic generated from automatic data.
    usice_inter : numpy.ndarray
        Ice type mosaic generated from USICE data.

    Returns:
    --------
    diff_us_aut : numpy.ndarray
        Difference between USICE and automatic ice type mosaics.
    res_aut : numpy.ndarray
        Automatic ice type mosaic.
    res_usnic : numpy.ndarray
        USICE ice type mosaic.
    mask_common : numpy.ndarray
        Mask of common areas between two mosaics.
    """
    # Create masks for automatic and USICE mosaics
    mask_aut = mosaic_aut >= 0
    mask_usnic = usice_inter >= 0
    
    # Calculate mask for common areas
    mask_common = mask_aut * mask_usnic

    # Apply common mask to automatic and USICE mosaics
    res_aut = mosaic_aut * mask_common
    res_usnic = usice_inter * mask_common
    
    # Replace NaN values with 0 in the USICE mosaic
    res_usnic = np.nan_to_num(res_usnic, nan=0)

    # Calculate the difference between USICE and automatic mosaics
    diff_us_aut = res_usnic - res_aut
    
    return diff_us_aut, res_aut, res_usnic, mask_common

def compute_stats_us_aut(man2aut, res_man, res_aut, mask_diff):
    """
    Compute various statistics comparing manual and automatic ice type mosaics.

    Parameters:
    -----------
    man2aut : numpy.ndarray
        Difference between manual and automatic ice type mosaics.
    res_man : numpy.ndarray
        Manual ice type mosaic.
    res_aut : numpy.ndarray
        Automatic ice type mosaic.
    mask_diff : numpy.ndarray
        Mask of areas where the two mosaics differ.

    Returns:
    --------
    result : dict
        Dictionary containing various statistics.
    """
    
    # Extract values where two mosaics differ
    m_man = res_man[mask_diff]
    m_aut = res_aut[mask_diff]
    
    # Calculate classification report
    report = skm.classification_report(m_man, m_aut, digits=3, output_dict=True)
    
    # Extract metrics from the report
    accuracy = report['accuracy']
    macro_avg_p = report['macro avg']['precision']
    macro_avg_r = report['macro avg']['recall']
    macro_avg_f = report['macro avg']['f1-score']
    
    weighted_avg_p = report['weighted avg']['precision']
    weighted_avg_r = report['weighted avg']['recall']
    weighted_avg_f = report['weighted avg']['f1-score']
    
    # Confusion matrix
    matrix = skm.confusion_matrix(m_man, m_aut)
    
    # Jaccard score
    jaccard_labels = skm.jaccard_score(m_man, m_aut, average=None)
    jaccard_avg = skm.jaccard_score(m_man, m_aut, average='weighted')
    
    # Cohen's kappa
    kappa = skm.cohen_kappa_score(m_man, m_aut)
    
    # Precision, recall, fscore, support
    p, r, f, s = skm.precision_recall_fscore_support(m_man, m_aut, average=None, warn_for=('precision', 'recall', 'f-score'))

    # Matthews correlation coefficient
    mcc = skm.matthews_corrcoef(m_man, m_aut)
    
    # Hamming loss
    hloss = skm.hamming_loss(m_man, m_aut)
    
    # Balanced accuracy
    b_acc = skm.balanced_accuracy_score(m_man, m_aut)
    
    # Count pixels in comparison, manual, and automatic images
    total_man = [np.count_nonzero(res_man[mask_diff] == i) for i in range(4)]
    total_aut = [np.count_nonzero(res_aut[mask_diff] == i) for i in range(4)]
    total = [np.count_nonzero(man2aut[mask_diff] == i) for i in range(-3, 4)]
    ind = list(range(-3, 4))
    
    # Prepare result dictionary
    result = dict(
        accuracy = accuracy,
        macro_precision = macro_avg_p,
        macro_recall = macro_avg_r,
        macro_f1_score = macro_avg_f,
        weighted_precision = weighted_avg_p,
        weighted_recall = weighted_avg_r,
        weighted_f1_score = weighted_avg_f,

        precision = p,
        recall = r,
        fscore = f,
        support = s,

        jaccard_labels = jaccard_labels,
        total = total,
        total_man = total_man,
        total_aut = total_aut,
        
        balanced_accuracy_score = b_acc,
        hamming_loss = hloss,
        cohen_kappa_score = kappa,
        jaccard_avg = jaccard_avg,

        matrix = matrix,
    )
    return result

def write_stats_day(datas, path_stats, filename):
    """
    Write statistics to a text file.

    Parameters:
    -----------
    datas : list
        List containing various statistics.
    path_stats : str
        Path to the directory where the statistics file will be saved.
    filename : str
        Name of the statistics file.

    Returns:
    --------
    None
    """
    with open(path_stats + filename + '.txt', "a") as file:
        
        for data in datas:
    
            if type(data) == float or isinstance(data, np.float64):
                file.write(str(data) + "\n")

            if type(data) == list:
                count = ';'.join(map(str, data))
                file.write(count + "\n")

            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    count = ';'.join(map(str, data))
                    file.write(count + "\n")

            if isinstance(data, np.ndarray):
                if data.ndim == 2:
                    file.write("Confusion matrix\n")
                    # Convert the 2D array to a string
                    rows = ["{};{};{};{}".format(i, j, k, l) for i, j, k, l in data]
                    conf = "\n".join(rows)
                    file.write(conf)
                    file.write("\n")


def image_render(year, month, day, path_img, man2aut, res_man, res_aut, land_mask, mask_diff):
    """
    Render and save an image showing the comparison between manual and automatic ice type mosaics.

    Parameters:
    -----------
    year : str
        Year.
    month : str
        Month.
    day : str
        Day.
    path_img : str
        Path to save the image.
    man2aut : numpy.ndarray
        Difference between manual and automatic ice type mosaics.
    res_man : numpy.ndarray
        Manual ice type mosaic.
    res_aut : numpy.ndarray
        Automatic ice type mosaic.
    land_mask : numpy.ndarray
        Mask of land areas.
    mask_diff : numpy.ndarray
        Mask of areas where the two mosaics differ.

    Returns:
    --------
    None
    """

    # Adapt array with values for no data and land
    img = man2aut
    img[~mask_diff] = -4
    img[land_mask] = -5

    # Manual
    img_man = res_man
    img_man[~mask_diff] = -1
    img_man[land_mask] = -2

    # Automatic
    img_aut = res_aut
    img_aut[~mask_diff] = -1
    img_aut[land_mask] = -2

    # Colormap for comparison
    cmap = plt.cm.colors.ListedColormap(['gray', 'white', '#b30727', '#e8775d', '#f0cab7', '#cfd9e8', '#b5cdf8', '#6485ec', '#384abf'])
    # Colormap for ice type (from H.Boulze)
    cmap_hugo = plt.cm.colors.ListedColormap(['whitesmoke', 'white', '#0064ff', '#aa28f0', '#ffff00', '#b46432'])

    # Normalization of ice comparison
    norm = plt.Normalize(-5, 4)
    img_norm = norm(img)
    img_ = cmap(img_norm)

    # Normalization of ice types
    norm2 = plt.Normalize(-2, 4)
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

    # Use tight_layout() only for certain axes
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])  # Adjust the rect values as needed
    
    cbar_comp = plt.colorbar(im1, ax=ax1)

    cbar_comp.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['ground', 'no data', '-3', '-2', '-1', '0', '1', '2', '3']):
        cbar_comp.ax.text(1.3, (j + 0.5) / 9.0, lab, ha='left', va='center', fontsize='small')

    cbaxes = fig.add_axes([0.5, 0.62, 0.4, 0.02])
    cbar = plt.colorbar(im2, ax=[ax2, ax3], orientation='horizontal', cax=cbaxes)

    cbar.ax.get_xaxis().set_ticks([])
    for j, lab in enumerate(['Ground', 'No Data', 'Ice free', 'Young Ice', 'First Year Ice', 'Multi Year Ice']):
        cbar.ax.text((j + 0.5) / 6.0, .5, lab, ha='center', va='center', fontsize='small')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    plt.savefig(path_img + "map_" + year + month + day + ".png", dpi=300, bbox_inches='tight')


def nic_nersc_day(year, month, day, path_nic, path_nersc, path_stats):
    """
    Process NIC and NERSC data for a given day, compute statistics, and render images.

    Parameters:
    -----------
    year : str
        Year.
    month : str
        Month.
    day : str
        Day.
    path_nic : str
        Path to NIC data.
    path_nersc : str
        Path to NERSC data.
    path_stats : str
        Path to save the statistics.

    Returns:
    --------
    None
    """
    ofilename = f'{path_stats}/stats_{year[2:4]}{month}{day}.npz'
    if os.path.exists(ofilename):
        return

    shapefile = path_nic + 'ARCTIC' + year[2:4] + month + day + '.shp'

    # Define raster parameters
    x_ul = -871516.0
    nx = 2800
    dx = 1000.0
    y_ul = 57017.0
    ny = 2500
    dy = -1000.0
    srs = '+proj=stere +lat_0=90.0 +lon_0=0.0 +lat_ts=90.0 +R=6.371e+06 +units=m +no_defs'

    # Rasterize the manual ice chart
    ds = get_gdal_dataset(x_ul, nx, dx, y_ul, ny, dy, srs, gdal.GDT_Int16)
    polyindex_arr, icecodes = rasterize_icehart(shapefile, ds)
    map_ice = ice_type_map(polyindex_arr, icecodes)

    # Mosaic the automatic files
    aut_files = week_auto_files(year + '-' + month + '-' + day, path_nersc)
    if len(aut_files) == 0:
        print('No automatic files')
        np.savez(ofilename, none=None)
        return

    auto_mosaic = mosaic_auto_argmax(aut_files)

    with Dataset(aut_files[0]) as ds:
        land_mask = ds['ice_type'][0].filled(0) == -1

    # Make difference map
    diff_us_aut, res_aut, res_usnic, mask_common = is_difference(auto_mosaic, map_ice)

    # Compute all statistics
    result = compute_stats_us_aut(diff_us_aut, res_usnic, res_aut, mask_common)

    # Write statistics to a file
    np.savez(ofilename, **result)

    # Render image of maps
    image_render(year, month, day, path_stats, diff_us_aut, res_usnic, res_aut, land_mask, mask_common)


def nic_nersc_comparison(start_date, end_date, path_nic, path_nersc, path_stats):
    """
    Compare NIC and NERSC data for a range of dates, compute statistics, and render images.

    Parameters:
    -----------
    start_date : datetime.date
        Start date.
    end_date : datetime.date
        End date.
    path_nic : str
        Path to NIC data.
    path_nersc : str
        Path to NERSC data.
    path_stats : str
        Path to save the statistics.

    Returns:
    --------
    None
    """
    # Iterate through each date in the range
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y-%m-%d"))
        day = single_date.strftime("%d")
        month = single_date.strftime("%m")
        year = single_date.strftime("%Y")
        path_nic_day = path_nic + 'arctic' + year[2:] + month + day + '/'
        
        # Find manual ice chart file
        man_file = sorted(glob.glob(path_nic_day + 'ARCTIC' + year[2:4] + month + day + '.shp'))
        if len(man_file) == 1:
            # Process NIC and NERSC data for the day
            nic_nersc_day(year, month, day, path_nic_day, path_nersc, path_stats)
        else:
            print('Manual ice chart does not exist')

        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="start date of computation YYYY-mm-dd")
    parser.add_argument("end", help="end date of computation YYYY-mm-dd")
    parser.add_argument("path_man", help="path of manuals data /path/to/data/")
    parser.add_argument("path_aut", help="path of automatics data /path/to/data/")
    parser.add_argument("path_stats", help="path of statistics results and images /path/to/data/")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    nic_nersc_comparison(start_date, end_date, args.path_man, args.path_aut, args.path_stats)

if __name__ == "__main__":
    main()