#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

import os
import argparse
from datetime import date

#from validate_on_nic_DMI import ValidationNIC_DMI
#from validate_on_dmi_DMI import ValidationDMI_DMI
#from validate_on_osisaf_DMI import ValidationOSISAF_DMI

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("product", help="Name of product to validate", choices=['DMI', 'NERSC'])
    parser.add_argument("reference", help="Name of reference product", choices=['DMI', 'NIC', 'OSISAF'])
    parser.add_argument("start", help="start date of computation YYYY-mm-dd")
    parser.add_argument("end", help="end date of computation YYYY-mm-dd")
    parser.add_argument('-s', '--step', help="Subsampling of DMI ice chart", type=int, default=10)
    parser.add_argument('-c', '--cores', help="Parallel cores to use", type=int, default=5)
    args = parser.parse_args()

    args.start_date = date.fromisoformat(args.start)
    args.end_date = date.fromisoformat(args.end)

    args.aut_dir = os.getenv(f'{args.product}_PRODUCT_DIR')
    if args.aut_dir is None:
        raise ValueError(f'Environment variable {args.product}_PRODUCT_DIR pointing to product for validation is not set.')
    args.ref_dir = os.getenv(f'{args.reference}_REFERENCE_DIR')
    if args.ref_dir is None:
        raise ValueError(f'Environment variable {args.reference}_REFERENCE_DIR pointing to reference product is not set.')

    return args

    #parser.add_argument("dir_man", help="Path to manual ice charts")
    #parser.add_argument("dir_aut", help="Path to automatic ice charts")
    #parser.add_argument("dir_stats", help="Path to save statistics results and images")

    #vn = ValidationClass(args.dir_man, args.dir_aut, args.dir_stats, args.cores)
    #vn.step = args.step
    #vn.process_date_range(start_date, end_date)

if __name__ == "__main__":
    args = parse()
    print(args)
