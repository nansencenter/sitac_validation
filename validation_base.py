import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from sitacval import (
    daterange,
    weekrange,
    compute_sod_stats,
    compute_sic_stats,
    plot_sod_map,
    plot_sic_map,
    plot_difference,
)

class ValidationBase:
    def __init__(self, dir_man, dir_auto, dir_stats, cores, step):
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
        self.step = step
        os.makedirs(self.dir_stats, exist_ok=True)

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
        print(f'Processing {date}')
        man_file = self.find_manual_file(date)
        if not os.path.exists(man_file):
            print(f'    No reference file {os.path.basename(man_file)}')
            return

        aut_files = self.week_auto_files(date)
        if len(aut_files) == 0:
            print('    No auto files')
            return
        aut_ice_chart = self.get_aut_ice_chart(aut_files)
        man_ice_chart = self.get_man_ice_chart(man_file)

        diff, mask = self.get_difference(man_ice_chart, aut_ice_chart)

        self.save_stats(date, man_ice_chart, aut_ice_chart, mask)
        self.make_maps(date, man_ice_chart, aut_ice_chart, diff, mask)

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

    def save_sod_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        aut_sod = np.round(aut_ice_chart['sod'][mask['sod']]).astype(int)
        man_sod = np.round(man_ice_chart['sod'][mask['sod']]).astype(int)
        sod_stats = compute_sod_stats(man_sod, aut_sod, self.max_value['sod'])
        sod_stats['labels'] = self.labels
        sod_stats_filename = f'{self.dir_stats}/stats_sod_{date.strftime("%Y%m%d")}.npz'
        np.savez(sod_stats_filename, **sod_stats)
        print(f'    Save {os.path.basename(sod_stats_filename)}')

    def save_sic_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        sic_stats = compute_sic_stats(man_ice_chart['sic'], aut_ice_chart['sic'], mask['sic'])
        sic_stats_filename = f'{self.dir_stats}/stats_sic_{date.strftime("%Y%m%d")}.npz'
        np.savez(sic_stats_filename, **sic_stats)
        print(f'    Save {os.path.basename(sic_stats_filename)}')

    def make_sod_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sod_map(man_ice_chart['sod'], aut_ice_chart['landmask'], axs[0], f'{self.map_label_man}-SoD, {date.strftime("%Y-%m-%d")}', self.labels)
        plot_sod_map(aut_ice_chart['sod'], aut_ice_chart['landmask'], axs[1], f'{self.map_label_aut}-SoD', self.labels, shrink=0)
        plot_difference(diff['sod'], mask['sod'], aut_ice_chart['landmask'], axs[2], 'Difference')
        map_filename = f'{self.dir_stats}/map_sod_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    Save {os.path.basename(map_filename)}')

    def make_sic_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sic_map(man_ice_chart['sic'], aut_ice_chart['landmask'], axs[0], f'{self.map_label_man}-SIC, {date.strftime("%Y-%m-%d")}')
        plot_sic_map(aut_ice_chart['sic'], aut_ice_chart['landmask'], axs[1], f'{self.map_label_aut}-SIC', shrink=0)
        plot_difference(diff['sic'], mask['sic'], aut_ice_chart['landmask'], axs[2], 'Difference', factor=10)
        map_filename = f'{self.dir_stats}/map_sic_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    Save {os.path.basename(map_filename)}')
