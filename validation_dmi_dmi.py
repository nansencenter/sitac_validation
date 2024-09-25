from cartopy.crs import NorthPolarStereo
import numpy as np

from sitacval import *
from validation_nic_dmi import Validation_NIC_DMI


class Validation_DMI_DMI(Validation_NIC_DMI):
    map_label_aut = 'DMI-auto'
    map_label_man = 'DMI-manual'
    products = ['sic', 'sod', 'flz']
    max_value = {
        'sic': 100,
        'sod': 4,
        'flz': 3,
    }
    labels = {
        'sod':[
            "Young Ice",
            "Thin FY Ice",
            "Thick FY Ice",
            "Multi-Year Ice",
            "Glacier Ice",
        ],
        'flz': [
            "Small Floes",
            "Medium Floes",
            "Big Floes",
            "Vast and Giant Floes",
        ],
    }

    def get_aut_ice_chart(self, aut_files):
        """ Get averaged SIC and SOD from predicted ice chart on the grid of manual DMI ice chart """
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
            [aut_ice_chart['sic'], aut_ice_chart['sod'], aut_ice_chart['flz'], aut_ice_chart['landmask']],
            dst_crs, xc, yc)
        return {
            'sic': aut_arrays_pro[0],
            'sod': aut_arrays_pro[1],
            'flz': aut_arrays_pro[2],
            'landmask': aut_arrays_pro[3],
        }

    def get_man_ice_chart(self, man_file):
        """ Get SIC and SOD from manual DMI ice chart """
        ice_type_maping = get_ice_type_mapping()
        floe_size_maping = get_floe_size_mapping()
        ct,ca,sa,fa,cb,sb,fb,cc,sc,fc,ice_poly_id_grid = get_man_file(man_file)

        cam, cbm, ccm = correct_ca_cb_cc(ct, np.array(ca), np.array(cb), np.array(cc))
        sam, sbm, scm = convert_sigrid_codes(sa, sb, sc, ice_type_maping)
        fam, fbm, fcm = convert_sigrid_codes(fa, fb, fc, floe_size_maping)

        ice_type_fractions = get_ice_type_fractions_dmi(cam, cbm, ccm, sam, sbm, scm)
        floe_size_fractions = get_ice_type_fractions_dmi(cam, cbm, ccm, fam, fbm, fcm)

        sic_map = get_sic_map_dmi(ct, ice_poly_id_grid)
        sod_map = get_sod_map_dmi(ice_type_fractions, ice_poly_id_grid)
        flz_map = get_sod_map_dmi(floe_size_fractions, ice_poly_id_grid)
        return {
            'sic': sic_map,
            'sod': sod_map,
            'flz': flz_map,
        }

    def find_manual_file(self, date):
        """ Find netCDF with DMI ice chart """
        man_file = date.strftime(f'{self.dir_man}/%Y/%m/ice_conc_overview_greenland_%Y%m%d1200.nc')
        return man_file

    def save_stats(self, date, man_ice_chart, aut_ice_chart, mask):
        self.save_sic_stats(date, man_ice_chart, aut_ice_chart, mask)
        self.save_sod_stats(date, man_ice_chart, aut_ice_chart, mask, 'sod')
        self.save_sod_stats(date, man_ice_chart, aut_ice_chart, mask, 'flz')

    def make_maps(self, date, man_ice_chart, aut_ice_chart, diff, mask):
        self.make_sic_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
        self.make_sod_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
        self.make_flz_maps(date, man_ice_chart, aut_ice_chart, diff, mask)
