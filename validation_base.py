import os
from multiprocessing import Pool

import numpy as np

from sitacval import daterange, weekrange

class ValidationBase:
    def __init__(self, dir_man, dir_auto, dir_stats, cores):
        """
        Parameters:
        -----------
        dir_man : str
            Path to NIC data.
        dir_auto : str
            Path to automatic ice charts
        dir_stats : str
            Path to save the statistics.
        cores : int
            Number of cores to use in parallel

        """
        self.dir_man = dir_man
        self.dir_auto = dir_auto
        self.dir_stats = dir_stats
        self.cores = cores

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

    def process_date(self, date):
        man_file = self.find_manual_file(date)
        if not os.path.exists(man_file):
            print(f'No manual {man_file} file for', date)
            return

        aut_files = self.week_auto_files(date)
        if len(aut_files) == 0:
            print('No auto files for ', date, man_file)
            return
        print('Processing ', date, man_file)
        aut_ice_chart = self.get_aut_ice_chart(aut_files)
        man_ice_chart = self.get_man_ice_chart(man_file)

        diff, mask = self.get_difference(man_ice_chart, aut_ice_chart)

        self.save_stats(date, man_ice_chart, aut_ice_chart, mask)
        self.make_maps(date, man_ice_chart, aut_ice_chart, diff, mask)

    def find_manual_file(self, date):
        shapefile = f'{self.dir_man}/arctic{date.strftime("%y%m%d")}/ARCTIC{date.strftime("%y%m%d")}.shp'
        return shapefile

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
        with Pool(self.cores) as p:
            p.map(self.process_date, daterange(start_date, end_date))

