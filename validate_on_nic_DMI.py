#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

from datetime import date
import argparse

from cmocean import cm as cmo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from netCDF4 import Dataset
from scipy.stats import pearsonr

from sitacval import ValidationNIC, get_gdal_dataset, rasterize_icehart, compute_stats, gdal

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

def get_ice_type_fractions(icecodes, CA, CB, CC, SA, SB, SC):
    ice_type_fractions = np.zeros((len(icecodes), 7))
    ice_type_fractions[range(len(icecodes)), SA] += CA
    ice_type_fractions[range(len(icecodes)), SB] += CB
    ice_type_fractions[range(len(icecodes)), SC] += CC
    ice_type_fractions[:, 1] = 100 - ice_type_fractions[:, 2:].sum(axis=1)
    return ice_type_fractions

def get_sod_sic_maps(polyindex_arr, icecodes, ice_type_fractions, CT):
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
    ice_type_fractions = get_ice_type_fractions(icecodes, CA, CB, CC, SA, SB, SC)
    sod_nic, sic_nic = get_sod_sic_maps(polyindex_arr, icecodes, ice_type_fractions, CT)
    return sod_nic, sic_nic

def read_dmi_ice_chart(dmi_file, step):
    with Dataset(dmi_file) as ds:
        sod_dmi = ds['sod'][0, ::step, ::step].astype(float).filled(np.nan) + 1
        sic_dmi = ds['sic'][0, ::step, ::step].astype(float).filled(np.nan)
        flg_dmi = ds['status_flag'][0, ::step, ::step]

    lnd_dmi = (flg_dmi & 64) > 0
    sic_dmi[lnd_dmi] = -1
    sod_dmi[sic_dmi < 15] = 0
    sod_dmi[lnd_dmi] = -1

    return sod_dmi, sic_dmi, lnd_dmi

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

class ValidationNIC_DMI(ValidationNIC):
    products = ['sod', 'sic']
    max_value = {'sod': 5, 'sic': 100}
    dir_auto_format = 'dmi_asip_seaice_mosaic_arc_l3_%Y%m%d.nc'
    labels = [
        "Ice Free",
        "Young Ice",
        "Thin FY Ice",
        "Thick FY Ice",
        "Multi-Year Ice",
        "Glacier Ice",
    ]

    def get_aut_ice_shart(self, aut_files):
        sods, sics =[], []
        for aut_file in aut_files:
            print('Reading automatic ice chart from ', aut_file)
            sod_dmi, sic_dmi, lnd_dmi = read_dmi_ice_chart(aut_file, self.step)
            sods.append(sod_dmi)
            sics.append(sic_dmi)
        sics = np.dstack(sics)
        sods = np.dstack(sods)
        return {
            'sod': np.nanmedian(sods, axis=2),
            'sic': np.nanmedian(sics, axis=2),
            'landmask': lnd_dmi
        }

    def get_man_ice_shart(self, shapefile):
        sod_nic, sic_nic = read_nic_icechart(shapefile, self.step)
        return {
            'sod': sod_nic,
            'sic': sic_nic,
        }

    def save_stats(self, date, man_ice_shart, aut_ice_shart, mask):
        aut_sod = np.round(aut_ice_shart['sod'][mask['sod']]).astype(int)
        man_sod = np.round(man_ice_shart['sod'][mask['sod']]).astype(int)
        stats = compute_stats(man_sod, aut_sod, self.max_value['sod'])
        sic_stats = compute_sic_stats(man_ice_shart['sic'], aut_ice_shart['sic'], mask['sic'])
        stats.update(sic_stats)
        stats['labels'] = self.labels
        stats_filename = f'{self.dir_stats}/stats_{date.strftime("%Y%m%d")}.npz'
        np.savez(stats_filename, **stats)
        print(stats_filename)

    def make_maps(self, date, man_ice_shart, aut_ice_shart, diff, mask):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sod_map(man_ice_shart['sod'], aut_ice_shart['landmask'], axs[0], f'NIC SoD, {date.strftime("%Y-%m-%d")}', self.labels)
        plot_sod_map(aut_ice_shart['sod'], aut_ice_shart['landmask'], axs[1], 'DMI SoD chart', self.labels, shrink=0)
        plot_difference(diff['sod'], mask['sod'], aut_ice_shart['landmask'], axs[2], 'Difference')
        map_filename = f'{self.dir_stats}/map_sod_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(map_filename)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sic_map(man_ice_shart['sic'], aut_ice_shart['landmask'], axs[0], f'NIC SIC, {date.strftime("%Y-%m-%d")}')
        plot_sic_map(aut_ice_shart['sic'], aut_ice_shart['landmask'], axs[1], 'DMI SIC', shrink=0)
        plot_difference(diff['sic'], mask['sic'], aut_ice_shart['landmask'], axs[2], 'Difference', factor=10)
        map_filename = f'{self.dir_stats}/map_sic_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(map_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="start date of computation YYYY-mm-dd")
    parser.add_argument("end", help="end date of computation YYYY-mm-dd")
    parser.add_argument("dir_man", help="Path to manual ice charts")
    parser.add_argument("dir_aut", help="Path to automatic ice charts")
    parser.add_argument("dir_stats", help="Path to save statistics results and images")
    parser.add_argument('-s', '--step', help="Subsampling of DMI ice chart", type=int, default=10)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    vn = ValidationNIC_DMI(args.dir_man, args.dir_aut, args.dir_stats)
    vn.step = args.step
    vn.process_date_range(start_date, end_date)

if __name__ == "__main__":
    main()