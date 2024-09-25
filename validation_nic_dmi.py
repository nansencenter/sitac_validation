import numpy as np
import matplotlib.pyplot as plt

from sitacval import *
from validation_base import ValidationBase


class Validation_NIC_DMI(ValidationBase):
    products = ['sod', 'sic']
    max_value = {'sod': 4, 'sic': 100}
    dir_auto_format = 'dmi_asip_seaice_mosaic_arc_l3_%Y%m%d.nc'
    labels = {
        'sod':[
            "Young Ice",
            "Thin FY Ice",
            "Thick FY Ice",
            "Multi-Year Ice",
            "Glacier Ice",
        ]
    }
    map_label_aut = 'DMI-auto'
    map_label_man = 'NIC-manual'

    def find_manual_file(self, date):
        """ Find shapefile with NIC ice chart """
        shapefile = f'{self.dir_man}/arctic{date.strftime("%y%m%d")}/ARCTIC{date.strftime("%y%m%d")}.shp'
        return shapefile

    def get_aut_ice_chart(self, aut_files):
        """ Get averaged predicted ice chart from several input netDCF files """
        sics, sods, flzs = [], [], []
        for aut_file in aut_files:
            sic_dmi, sod_dmi, flz_dmi, lnd_dmi, xc, yc = read_dmi_ice_chart(aut_file, self.step)
            sods.append(sod_dmi)
            sics.append(sic_dmi)
            flzs.append(flz_dmi)
        sics, sods, flzs = [np.nanmedian(np.dstack(i), axis=2) for i in [sics, sods, flzs]]
        return {
            'sic': sics,
            'sod': sods,
            'flz': flzs,
            'landmask': lnd_dmi,
            'xc': xc,
            'yc': yc,
        }

    def get_man_ice_chart(self, shapefile):
        """ Get reference ice chart from NIC """
        sod_nic, sic_nic = read_nic_icechart(shapefile, self.step)
        return {
            'sod': sod_nic,
            'sic': sic_nic,
        }

    def save_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        self.save_sic_stats(date, man_ice_chart, aut_ice_chart, mask)
        self.save_sod_stats(date, man_ice_chart, aut_ice_chart, mask, 'sod')

    def make_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        self.make_sod_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
        self.make_sic_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
