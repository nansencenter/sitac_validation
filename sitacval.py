from datetime import timedelta
import os

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
    matrix = skm.confusion_matrix(man_pixels, aut_pixels)

    # Jaccard score
    jaccard_labels = skm.jaccard_score(man_pixels, aut_pixels, average=None)
    jaccard_avg = skm.jaccard_score(man_pixels, aut_pixels, average='weighted')

    # Cohen's kappa
    kappa = skm.cohen_kappa_score(man_pixels, aut_pixels)

    # Precision, recall, fscore, support
    p, r, f, s = skm.precision_recall_fscore_support(man_pixels, aut_pixels, average=None, warn_for=('precision', 'recall', 'f-score'))

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

class ValidationNIC:
    def __init__(self, dir_man, dir_auto, dir_stats):
        """
        Parameters:
        -----------
        dir_man : str
            Path to NIC data.
        dir_auto : str
            Path to automatic ice charts
        dir_stats : str
            Path to save the statistics.

        """
        self.dir_man = dir_man
        self.dir_auto = dir_auto
        self.dir_stats = dir_stats

    def get_difference(self, man_chart, aut_chart):
        """
        Calculate the difference between two ice charts

        Parameters:
        -----------
        man_chart : numpy.ndarray
            Ice type mosaic generated from NIC data.
        aut_chart : numpy.ndarray
            Ice type mosaic generated from automatic data.

        Returns:
        --------
        diff : numpy.ndarray
            Difference between man_chart and aut_chart
        res_man : numpy.ndarray
            NIC ice type mosaic.
        res_aut : numpy.ndarray
            Automatic ice type mosaic.
        mask_common : numpy.ndarray
            Mask of common areas between two mosaics.

        """
        diff = {}
        mask = {}
        for prod in self.products:
            # Calculate mask
            mask[prod] = (
                (man_chart[prod] >= 0) *
                (aut_chart[prod] >= 0) *
                np.isfinite(man_chart[prod] * aut_chart[prod])
            )

            # Calculate the difference between NIC and automatic mosaics
            diff[prod] = aut_chart[prod] - man_chart[prod]
            diff[prod][~mask[prod]] = 0

        return diff, mask

    def week_auto_files(self, end_date):
        """
        Get the list of automatic files for the week ending at str_date.

        Parameters:
        -----------
        end_date : str
            Date of NIC shapefile (end of period)

        Returns:
        --------
        aut_files : list
            List of automatic files for the week ending at end_date.
        """
        aut_files = []
        for single_date in weekrange(end_date):
            filename = f'{self.dir_auto}/{single_date.strftime(self.dir_auto_format)}'
            if os.path.exists(filename):
                aut_files.append(filename)
        return aut_files

    def process_day(self, date, shapefile):
        aut_files = self.week_auto_files(date)
        if len(aut_files) == 0:
            print('No input file for ', date, shapefile)
            return
        print('Processing ', date, shapefile)
        aut_ice_shart = self.get_aut_ice_shart(aut_files)
        man_ice_shart = self.get_man_ice_shart(shapefile)

        diff, mask = self.get_difference(man_ice_shart, aut_ice_shart)

        self.save_stats(date, man_ice_shart, aut_ice_shart, mask)
        self.make_maps(date, man_ice_shart, aut_ice_shart, diff, mask)

    def process_date_range(self, start_date, end_date, ):
        """
        Compare NIC and automatic ice charts for a range of dates, compute statistics, and render images.

        Parameters:
        -----------
        start_date : datetime.date
            Start date.
        end_date : datetime.date
            End date.

        """
        # Iterate through each date in the range
        for date in daterange(start_date, end_date):
            shapefile = f'{self.dir_man}/arctic{date.strftime("%y%m%d")}/ARCTIC{date.strftime("%y%m%d")}.shp'
            if os.path.exists(shapefile):
                self.process_day(date, shapefile)


