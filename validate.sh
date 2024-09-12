export DMI_PRODUCT_DIR=/Data/sat/auxdata/ice_charts/dmi_asip_seaice_mosaic_arc_l3
export NERSC_PRODUCT_DIR=/Data/sat/auxdata/ice_charts/cmems_obs-si_arc_phy-icetype_nrt_L4-auto_P1D

export NIC_REFERENCE_DIR=/Data/sat/auxdata/ice_charts/NIC
export DMI_REFERENCE_DIR=/Data/sat/auxdata/ice_charts/cmems_obs-si_arc_physic_nrt_1km-grl_P1WT3D-m_202012
export OSISAF_REFERENCE_DIR=/Data/sim/data/OSISAF_ice_conc_amsr

# ./validate.py AUTOMATIC_PRODUCT REFERENCE_PRODUCT START_DATE END_DATE
./validate.py DMI NIC dmi_nic 2021-01-01 2021-03-31
./validate.py DMI DMI dmi_dmi 2021-01-01 2021-03-31
./validate.py DMI OSISAF dmi_osisaf 2021-01-01 2021-03-31