from datetime import timedelta

from cmocean import cm as cmo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

import numpy as np
from osgeo import gdal, osr, ogr
import sklearn.metrics as skm

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

def compute_stats(man_pixels, aut_pixels, max_val):
    # Calculate classification report
    report = skm.classification_report(man_pixels, aut_pixels, digits=3, output_dict=True)

    # Extract metrics from the report
    accuracy = report['accuracy']
    macro_avg_p = report['macro avg']['precision']
    macro_avg_r = report['macro avg']['recall']
    macro_avg_f = report['macro avg']['f1-score']

    weighted_avg_p = report['weighted avg']['precision']
    weighted_avg_r = report['weighted avg']['recall']
    weighted_avg_f = report['weighted avg']['f1-score']

    # Confusion matrix
    matrix = skm.confusion_matrix(man_pixels, aut_pixels, labels=range(max_val+1))
    matrix = np.where(matrix == 0, np.nan, matrix)

    # Jaccard score
    jaccard_labels = skm.jaccard_score(man_pixels, aut_pixels, average=None)
    jaccard_avg = skm.jaccard_score(man_pixels, aut_pixels, average='weighted')

    # Cohen's kappa
    kappa = skm.cohen_kappa_score(man_pixels, aut_pixels)

    # Precision, recall, fscore, support
    p, r, f, s = skm.precision_recall_fscore_support(
        man_pixels,
        aut_pixels,
        average=None,
        labels=range(6)
    )
    p, r, f, s = [np.where(j == 0, np.nan, j) for j in [p, r, f, s]]


    # Matthews correlation coefficient
    mcc = skm.matthews_corrcoef(man_pixels, aut_pixels)

    # Hamming loss
    hloss = skm.hamming_loss(man_pixels, aut_pixels)

    # Balanced accuracy
    b_acc = skm.balanced_accuracy_score(man_pixels, aut_pixels)

    ## Count pixels in comparison, manual, and automatic images
    total_man = [np.count_nonzero(man_pixels == i) for i in range(max_val+1)]
    total_aut = [np.count_nonzero(aut_pixels == i) for i in range(max_val+1)]
    total = [np.count_nonzero((aut_pixels - man_pixels) == i) for i in range(-max_val, max_val+1)]

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
        matthews_corrcoef = mcc,

        matrix = matrix,
    )
    return result

def get_dmi_dataset(step=10):
    srs = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
    x_ul = -3849750.0
    y_ul = 5849750.0
    nx = 15200 // step
    ny = 22400 // step
    dx = 500. * step
    dy = -500. * step
    return get_gdal_dataset(x_ul, nx, dx, y_ul, ny, dy, srs, gdal.GDT_Int16)

def get_ice_type_mapping():
    ice_type_maping = np.zeros(100, int)
    # water
    ice_type_maping[0] = 1
    #0 - Young Ice (81, 82, 83, 84, 85)
    ice_type_maping[81] = 2
    ice_type_maping[82] = 2
    ice_type_maping[83] = 2
    ice_type_maping[84] = 2
    ice_type_maping[85] = 2
    #1 - Thin FY Ice (87, 88, 89)
    ice_type_maping[87] = 3
    ice_type_maping[88] = 3
    ice_type_maping[89] = 3
    #2 - Thick FY Ice (86, 91, 93)
    ice_type_maping[86] = 4
    ice_type_maping[91] = 4
    ice_type_maping[93] = 4
    #3 - MY Ice (95, 96, 97)
    ice_type_maping[95] = 5
    ice_type_maping[96] = 5
    ice_type_maping[97] = 5
    #4 - Glacier Ice (98)
    ice_type_maping[98] = 6
    return ice_type_maping

def get_ct_ca_cb_cc(icecodes):
    CT, CA, CB, CC = icecodes[:, 1:5].T.astype(int)
    CA[CA == -9] = CT[CA == -9]
    CB[CB == -9] = 0
    CC[CC == -9] = 0
    return CT, CA, CB, CC

def get_sa_sb_sc(icecodes, ice_type_maping):
    SA_SB_SC = []
    for i in [5, 6, 7]:
        icecodes_S = icecodes[:, i].astype(int)
        icecodes_S[icecodes_S  == -9] = 99
        mapped_S = ice_type_maping[icecodes_S]
        SA_SB_SC.append(mapped_S)
    return SA_SB_SC

def get_ice_type_fractions_nic(icecodes, CA, CB, CC, SA, SB, SC):
    ice_type_fractions = np.zeros((len(icecodes), 7))
    ice_type_fractions[range(len(icecodes)), SA] += CA
    ice_type_fractions[range(len(icecodes)), SB] += CB
    ice_type_fractions[range(len(icecodes)), SC] += CC
    ice_type_fractions[:, 1] = 100 - ice_type_fractions[:, 2:].sum(axis=1)
    return ice_type_fractions

def get_sod_sic_maps_nic(polyindex_arr, icecodes, ice_type_fractions, CT):
    icecodes_i = icecodes[:, 0].astype(int)
    sod_poly = np.zeros(polyindex_arr.max() + 1)
    sod_poly[icecodes_i] = np.argmax(ice_type_fractions, axis=1)
    sod_map = sod_poly[polyindex_arr] - 1

    sic_poly = np.zeros(polyindex_arr.max() + 1) - 1
    sic_poly[icecodes_i] = CT #ice_type_fractions[:, 1]
    sic_map = sic_poly[polyindex_arr]
    return sod_map, sic_map

def read_nic_icechart(nic_file, step):
    ds = get_dmi_dataset(step)
    polyindex_arr, icecodes = rasterize_icehart(nic_file, ds)
    ice_type_maping = get_ice_type_mapping()
    CT, CA, CB, CC = get_ct_ca_cb_cc(icecodes)
    SA, SB, SC = get_sa_sb_sc(icecodes, ice_type_maping)
    ice_type_fractions = get_ice_type_fractions_nic(icecodes, CA, CB, CC, SA, SB, SC)
    sod_nic, sic_nic = get_sod_sic_maps_nic(polyindex_arr, icecodes, ice_type_fractions, CT)
    return sod_nic, sic_nic

def read_dmi_ice_chart(dmi_file, step):
    with Dataset(dmi_file) as ds:
        sod_dmi = ds['sod'][0, ::step, ::step].astype(float).filled(np.nan) + 1
        sic_dmi = ds['sic'][0, ::step, ::step].astype(float).filled(np.nan)
        flg_dmi = ds['status_flag'][0, ::step, ::step]
        xc = ds['xc'][::step]
        yc = ds['yc'][::step]

    lnd_dmi = (flg_dmi & 64) > 0
    sic_dmi[lnd_dmi] = -1
    sod_dmi[sic_dmi < 15] = 0
    sod_dmi[lnd_dmi] = -1

    return sod_dmi, sic_dmi, lnd_dmi, xc, yc

def plot_difference(diff_array, mask_common, land_mask, ax, title, shrink=0.5, factor=1.):
    cowa = cm.get_cmap('coolwarm', 7)
    coolwarm_colors = [colors.rgb2hex(cowa(i)) for i in range(0, cowa.N)]
    cmap_diff = plt.cm.colors.ListedColormap(['gray', 'white'] + coolwarm_colors)
    diff_array = np.array(diff_array) / factor
    diff_array[~mask_common] = -4
    diff_array[land_mask] = -5
    tick_labels = np.arange(-3., 4.)
    tick_labels *= factor

    imsh = ax.imshow(diff_array, clim=[-5, 3], cmap=cmap_diff, interpolation='nearest')
    cbar = plt.colorbar(imsh, ax=ax, shrink=shrink)
    cbar.ax.yaxis.set_ticks(np.linspace(-4.5, 2.5, 9), ['land', 'no data'] + list(tick_labels.astype(int)))
    ax.set_title(title)

def plot_sod_map(sod_array, land_mask, ax, title, labels, shrink=0.5):
    map_array = np.array(sod_array)
    map_array[np.isnan(map_array)] = -2
    map_array[land_mask] = -1
    cmap_hugo = plt.cm.colors.ListedColormap(['white', '#bbb', '#0064ff', '#aa28f0', '#ffff00', '#ca0', '#e54', '#500'])

    im10 = ax.imshow(map_array, interpolation='nearest', cmap=cmap_hugo, clim=[-2, 5])
    if shrink > 0:
        cbar = plt.colorbar(im10, ax=ax, shrink=shrink)
        cbar.ax.yaxis.set_ticks(np.linspace(-1.5, 4.5, 8), ['No Data', 'Land'] + labels)
    ax.set_title(title)

def plot_sic_map(sic_array, land_mask, ax, title, shrink=0.5):
    map_array = np.array(sic_array)
    map_array[np.isnan(map_array)] = -1
    map_array[land_mask] = -2
    ice_colors = [colors.rgb2hex(i) for i in cmo.ice.resampled(101)(np.arange(0, 101))]
    sic_cmap = plt.cm.colors.ListedColormap(['#ccb', '#ffe'] + ice_colors)
    im10 = ax.imshow(map_array, interpolation='nearest', cmap=sic_cmap, clim=[-2, 100])
    if shrink > 0:
        cbar = plt.colorbar(im10, ax=ax, shrink=shrink)
    ax.set_title(title)

def compute_sic_stats(man_sic, aut_sic, mask_sic):
    man_sic_ = man_sic[mask_sic]
    aut_sic_ = aut_sic[mask_sic]

    man_sic_bins = np.unique(man_sic_)
    aut_sic_avgs = []
    aut_sic_stds = []
    for man_sic_bin in man_sic_bins:
        gpi = man_sic_ == man_sic_bin
        aut_sic_avgs.append(aut_sic_[gpi].mean())
        aut_sic_stds.append(aut_sic_[gpi].std())

    aut_sic_avgs = np.array(aut_sic_avgs)
    aut_sic_stds = np.array(aut_sic_stds)
    return {
        'all_sic_pearsonr': pearsonr(man_sic_, aut_sic_)[0],
        'avg_sic_pearsonr': pearsonr(man_sic_bins, aut_sic_avgs)[0],
    }

# DMI reference ice chart
def get_man_file(path):
    with Dataset(path) as ds:
        ct = ds['CT'][0].astype(int).filled(0)
        ca = ds['CA'][0].astype(int).filled(0)
        cb = ds['CB'][0].astype(int).filled(0)
        cc = ds['CC'][0].astype(int).filled(0)
        sa = ds['SA'][0].astype(int).filled(0)
        sb = ds['SB'][0].astype(int).filled(0)
        sc = ds['SC'][0].astype(int).filled(0)
        ice_poly_id_grid = ds['ice_poly_id_grid'][0, ::-1]
    return ct,ca,sa,cb,sb,cc,sc,ice_poly_id_grid

def get_ice_type_fractions_dmi(icecodes, CA, CB, CC, SA, SB, SC):
    ice_type_fractions = np.zeros((len(icecodes), 7))
    ice_type_fractions[range(len(icecodes)), SA] += CA
    ice_type_fractions[range(len(icecodes)), SB] += CB
    ice_type_fractions[range(len(icecodes)), SC] += CC
    ice_type_fractions[:, 1] = 100 - ice_type_fractions[:, 2:].sum(axis=1)
    return ice_type_fractions

def correct_ca_cb_cc(CT, CA, CB, CC):
    CA[CA == -9] = CT[CA == -9]
    CB[CB == -9] = 0
    CC[CC == -9] = 0
    return CA, CB, CC

def correct_sa_sb_sc(SA, SB, SC, ice_type_maping):
    SA_SB_SC = []
    for s in [SA, SB, SC]:
        s[s  == -9] = 99
        SA_SB_SC.append(ice_type_maping[s])
    return SA_SB_SC

def get_sod_sic_maps_dmi(ice_type_fractions, ice_poly_id_grid, ct):
    sod = np.argmax(ice_type_fractions, axis=1)
    ice_poly_id_grid_int = ice_poly_id_grid.filled(0).astype(int)
    sic_map = ct[ice_poly_id_grid_int].astype(float)
    sic_map[ice_poly_id_grid.mask] = np.nan
    sod_map = sod[ice_poly_id_grid_int].astype(float) - 1
    sod_map[ice_poly_id_grid.mask] = np.nan
    return sod_map, sic_map

def reproject(src_crs, src_x, src_y, src_arrays, dst_crs, dst_x, dst_y):
    dst_x_grd, dst_y_grd = np.meshgrid(dst_x, dst_y)
    dst_x_grd_pro, dst_y_grd_pro, _ = src_crs.transform_points(dst_crs, dst_x_grd.flatten(), dst_y_grd.flatten()).T
    dst_arrays = []
    for src_array in src_arrays:
        rgi = RegularGridInterpolator((src_y, src_x), src_array, method='nearest', bounds_error=False)
        dst_array = rgi((dst_y_grd_pro, dst_x_grd_pro))
        dst_arrays.append(dst_array.reshape(dst_x_grd.shape))
    return dst_arrays
