import numpy as np
import matplotlib.pyplot as plt

from sitacval import *
from validation_base import ValidationBase


class ValidationNIC_DMI(ValidationBase):
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
    map_label_aut = 'DMI-auto'
    map_label_man = 'NIC-manual'

    def get_aut_ice_chart(self, aut_files):
        sods, sics =[], []
        for aut_file in aut_files:
            print('Reading automatic ice chart from ', aut_file)
            sod_dmi, sic_dmi, lnd_dmi, xc, yc = read_dmi_ice_chart(aut_file, self.step)
            sods.append(sod_dmi)
            sics.append(sic_dmi)
        sics = np.dstack(sics)
        sods = np.dstack(sods)
        return {
            'sod': np.nanmedian(sods, axis=2),
            'sic': np.nanmedian(sics, axis=2),
            'landmask': lnd_dmi,
            'xc': xc,
            'yc': yc,
        }

    def get_man_ice_chart(self, shapefile):
        sod_nic, sic_nic = read_nic_icechart(shapefile, self.step)
        return {
            'sod': sod_nic,
            'sic': sic_nic,
        }

    def save_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        aut_sod = np.round(aut_ice_chart['sod'][mask['sod']]).astype(int)
        man_sod = np.round(man_ice_chart['sod'][mask['sod']]).astype(int)
        stats = compute_stats(man_sod, aut_sod, self.max_value['sod'])
        sic_stats = compute_sic_stats(man_ice_chart['sic'], aut_ice_chart['sic'], mask['sic'])
        stats.update(sic_stats)
        stats['labels'] = self.labels
        stats_filename = f'{self.dir_stats}/stats_{date.strftime("%Y%m%d")}.npz'
        np.savez(stats_filename, **stats)
        print(stats_filename)

    def make_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sod_map(man_ice_chart['sod'], aut_ice_chart['landmask'], axs[0], f'{self.map_label_man}-SoD, {date.strftime("%Y-%m-%d")}', self.labels)
        plot_sod_map(aut_ice_chart['sod'], aut_ice_chart['landmask'], axs[1], f'{self.map_label_aut}-SoD', self.labels, shrink=0)
        plot_difference(diff['sod'], mask['sod'], aut_ice_chart['landmask'], axs[2], 'Difference')
        map_filename = f'{self.dir_stats}/map_sod_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(map_filename)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_sic_map(man_ice_chart['sic'], aut_ice_chart['landmask'], axs[0], f'{self.map_label_man}-SIC, {date.strftime("%Y-%m-%d")}')
        plot_sic_map(aut_ice_chart['sic'], aut_ice_chart['landmask'], axs[1], f'{self.map_label_aut}-SIC', shrink=0)
        plot_difference(diff['sic'], mask['sic'], aut_ice_chart['landmask'], axs[2], 'Difference', factor=10)
        map_filename = f'{self.dir_stats}/map_sic_{date.strftime("%Y%m%d")}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(map_filename)
