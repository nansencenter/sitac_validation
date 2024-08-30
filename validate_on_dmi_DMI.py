#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

from cartopy.crs import NorthPolarStereo
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from validate_on_nic_DMI import ValidationNIC_DMI, parse_and_run

def get_ice_type_mapping():
    ice_type_maping = np.zeros(100, int)
    #1 water
    ice_type_maping[0] = 1
    #2 - Young Ice (81, 82, 83, 84, 85)
    ice_type_maping[81] = 2
    ice_type_maping[82] = 2
    ice_type_maping[83] = 2
    ice_type_maping[84] = 2
    ice_type_maping[85] = 2
    #3 - Thin FY Ice (87, 88, 89)
    ice_type_maping[87] = 3
    ice_type_maping[88] = 3
    ice_type_maping[89] = 3
    #4 - Thick FY Ice (86, 91, 93)
    ice_type_maping[86] = 4
    ice_type_maping[91] = 4
    ice_type_maping[93] = 4
    #5 - MY Ice (95, 96, 97)
    ice_type_maping[95] = 5
    ice_type_maping[96] = 5
    ice_type_maping[97] = 5
    #6 - Glacier Ice (98)
    ice_type_maping[98] = 6
    return ice_type_maping

def get_ice_type_fractions(icecodes, CA, CB, CC, SA, SB, SC):
    ice_type_fractions = np.zeros((len(icecodes), 7))
    ice_type_fractions[range(len(icecodes)), SA] += CA
    ice_type_fractions[range(len(icecodes)), SB] += CB
    ice_type_fractions[range(len(icecodes)), SC] += CC
    ice_type_fractions[:, 1] = 100 - ice_type_fractions[:, 2:].sum(axis=1)
    return ice_type_fractions

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

def get_sod_sic_maps(ice_type_fractions, ice_poly_id_grid, ct):
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

class ValidationDMI_DMI(ValidationNIC_DMI):
    map_label_aut = 'DMI-auto'
    map_label_man = 'DMI-manual'

    def get_aut_ice_chart(self, aut_files):
        aut_ice_chart = super().get_aut_ice_chart(aut_files)
        # MAN ICE CHART
        #		crs:proj4_string = " +proj=stere +lon_0=-45 +lat_ts=90 +lat_0=90 +a=6371000 +b=6371000" ;
        dst_crs = NorthPolarStereo(true_scale_latitude=90, central_longitude=-45)
        # AUTO ICE CHART
        # 		crs:proj4_string = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs" ;
        src_crs = NorthPolarStereo(true_scale_latitude=70, central_longitude=-45)
        x_min = -923125.7
        x_max = 1307874.2
        x_size = 2232

        y_min = -529677.75
        y_max = -3846677.8
        y_size = 3318

        xc = np.linspace(x_min, x_max, x_size)
        yc = np.linspace(y_min, y_max, y_size)
        aut_arrays_pro = reproject(
            src_crs,
            aut_ice_chart['xc'],
            aut_ice_chart['yc'],
            [aut_ice_chart['sic'], aut_ice_chart['sod'], aut_ice_chart['landmask']],
            dst_crs, xc, yc)
        return {
            'sic': aut_arrays_pro[0],
            'sod': aut_arrays_pro[1],
            'landmask': aut_arrays_pro[2],
        }

    def get_man_ice_chart(self, man_file):
        ice_type_maping = get_ice_type_mapping()
        ct,ca,sa,cb,sb,cc,sc,ice_poly_id_grid = get_man_file(man_file)
        cam, cbm, ccm = correct_ca_cb_cc(ct, np.array(ca), np.array(cb), np.array(cc))
        sam, sbm, scm = correct_sa_sb_sc(sa, sb, sc, ice_type_maping)
        ice_type_fractions = get_ice_type_fractions(cam, cam, cbm, ccm, sam, sbm, scm)
        sod_map, sic_map = get_sod_sic_maps(ice_type_fractions, ice_poly_id_grid, ct)
        return {
            'sic': sic_map,
            'sod': sod_map,
        }

    def find_manual_file(self, date):
        man_file = date.strftime(f'{self.dir_man}/%Y/%m/ice_conc_overview_greenland_%Y%m%d1200.nc')
        return man_file


if __name__ == "__main__":
    parse_and_run(ValidationDMI_DMI)