#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

import os
import argparse
from datetime import date

from validation_nic_dmi import Validation_NIC_DMI
from validation_dmi_dmi import Validation_DMI_DMI
from validation_osisaf_dmi import Validation_OSISAF_DMI
from validation_nic_nersc import Validation_NIC_NERSC

VALIDATION_CLASSES = {
        'NIC_DMI': Validation_NIC_DMI,
        'DMI_DMI': Validation_DMI_DMI,
        'OSISAF_DMI': Validation_OSISAF_DMI,
        'NIC_NESRC': Validation_NIC_NERSC,
    }

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("product", help="Name of product to validate", choices=['DMI', 'NERSC'])
    parser.add_argument("reference", help="Name of reference product", choices=['DMI', 'NIC', 'OSISAF'])
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("start", help="start date of computation YYYY-mm-dd")
    parser.add_argument("end", help="end date of computation YYYY-mm-dd")
    parser.add_argument('-s', '--step', help="Subsampling of DMI ice chart", type=int, default=10)
    parser.add_argument('-c', '--cores', help="Parallel cores to use", type=int, default=5)
    args = parser.parse_args()

    args.start_date = date.fromisoformat(args.start)
    args.end_date = date.fromisoformat(args.end)

    args.ref_dir = os.getenv(f'{args.reference}_REFERENCE_DIR')
    if args.ref_dir is None:
        raise ValueError(f'Environment variable {args.reference}_REFERENCE_DIR pointing to reference product is not set.')
    args.aut_dir = os.getenv(f'{args.product}_PRODUCT_DIR')
    if args.aut_dir is None:
        raise ValueError(f'Environment variable {args.product}_PRODUCT_DIR pointing to product for validation is not set.')

    return args

if __name__ == "__main__":
    args = parse()
    ValidationClass = VALIDATION_CLASSES[f'{args.reference}_{args.product}']
    v = ValidationClass(args.ref_dir, args.aut_dir, args.out_dir, args.cores, args.step)
    v.process_date_range(args.start_date, args.end_date)