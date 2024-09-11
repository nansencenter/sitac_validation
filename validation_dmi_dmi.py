from cartopy.crs import NorthPolarStereo
import numpy as np

from sitacval import *
from validation_nic_dmi import Validation_NIC_DMI


class Validation_DMI_DMI(Validation_NIC_DMI):
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
        ice_type_fractions = get_ice_type_fractions_dmi(cam, cam, cbm, ccm, sam, sbm, scm)
        sod_map, sic_map = get_sod_sic_maps_dmi(ice_type_fractions, ice_poly_id_grid, ct)
        return {
            'sic': sic_map,
            'sod': sod_map,
        }

    def find_manual_file(self, date):
        man_file = date.strftime(f'{self.dir_man}/%Y/%m/ice_conc_overview_greenland_%Y%m%d1200.nc')
        return man_file
