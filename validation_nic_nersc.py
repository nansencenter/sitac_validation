#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

from datetime import date
from matplotlib.gridspec import GridSpec
import argparse
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from osgeo import gdal

from validation_base import ValidationBase
from sitacval import *


class Validation_NIC_NERSC(ValidationBase):
    products = ['sod']
    max_value = {'sod': 3}
    dir_auto_format = '%Y/%m/s1_icetype_mosaic_%Y%m%d0600.nc'
    labels = [
        "Ice Free",
        "Young Ice",
        "First-Year Ice",
        "Multi-Year Ice",
    ]

    def get_man_ice_chart(self, shapefile):
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
            return {'sod': map_ice}

    def get_aut_ice_chart(self, aut_files):
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

        with Dataset(aut_files[0]) as ds:
            land_mask = ds['ice_type'][0].filled(0) == -1
        return {'sod': max_prob_idx, 'landmask': land_mask}

    def save_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        stats = compute_stats(man_ice_chart['sod'][mask['sod']], aut_ice_chart['sod'][mask['sod']], self.max_value['sod'])
        stats['labels'] = self.labels
        stats_filename = f'{self.dir_stats}/stats_{date.strftime("%Y%m%d")}.npz'
        np.savez(stats_filename, **stats)
        print('    Save', os.path.basename(stats_filename))

    def image_render(self, date, man2aut, res_man, res_aut, land_mask, mask_diff):
        """
        Render and save an image showing the comparison between manual and automatic ice type mosaics.

        Parameters:
        -----------
        date : datetime
            Date of manual ice chart.
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
        fig.suptitle(f'SoD comparison {date.strftime("%Y-%m-%d")}', fontsize='x-large')

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
        map_filename = f'{self.dir_stats}/map_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print('    Save', os.path.basename(map_filename))

    def make_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        self.image_render(date, diff['sod'], man_ice_chart['sod'], aut_ice_chart['sod'], aut_ice_chart['landmask'], mask['sod'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="start date of computation YYYY-mm-dd")
    parser.add_argument("end", help="end date of computation YYYY-mm-dd")
    parser.add_argument("dir_man", help="Path to manual ice charts")
    parser.add_argument("dir_aut", help="Path to automatic ice charts")
    parser.add_argument("dir_stats", help="Path to save statistics results and images")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    vn = ValidationNIC_NERSC(args.dir_man, args.dir_aut, args.dir_stats)
    vn.process_date_range(start_date, end_date)

if __name__ == "__main__":
    main()