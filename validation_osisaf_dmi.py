import os
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from sitacval import read_dmi_ice_chart
from validation_dmi_dmi import Validation_DMI_DMI

class Validation_OSISAF_DMI(Validation_DMI_DMI):
    products = ['sic']
    map_label_aut = 'DMI-auto'
    map_label_man = 'OSI-SAF'

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
        auto_file = f'{self.dir_auto}/{end_date.strftime(self.dir_auto_format)}'
        if os.path.exists(auto_file):
            return [auto_file]
        else:
            return []

    def get_aut_ice_chart(self, aut_files):
        y_min, y_max, y_size = -5345000, 5845000,  1120
        man_y = np.linspace(y_max, y_min, y_size)
        x_min, x_max, x_size = -3845000, 3745000, 760
        man_x = np.linspace(x_min, x_max, x_size)

        _, sic_dmi, lnd_dmi, xc, yc = read_dmi_ice_chart(aut_files[0], self.step)

        dmi_arrays_pro = []
        for dmi_array in [sic_dmi, lnd_dmi]:
            rgi = RegularGridInterpolator((yc, xc), dmi_array, method='nearest', bounds_error=False)
            man_y_grd, man_x_grd = np.meshgrid(man_y, man_x, indexing='ij')
            dmi_array_pro = rgi((man_y_grd, man_x_grd))
            dmi_arrays_pro.append(dmi_array_pro)

        return {
            'sic': dmi_arrays_pro[0],
            'landmask': dmi_arrays_pro[1],
        }

    def get_man_ice_chart(self, man_file):
        with Dataset(man_file) as ds:
            sic_map = ds['ice_conc'][0, :, :]
        return {
            'sic': sic_map,
        }

    def find_manual_file(self, date):
        shapefile = f'{self.dir_man}/ice_conc_nh_polstere-100_amsr2_{date.strftime("%Y%m%d")}1200.nc'
        return shapefile

    def save_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        self.save_sic_stats(date, man_ice_chart, aut_ice_chart, mask)

    def make_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        self.make_sic_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
