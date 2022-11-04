#!/bin/bash

mkdir -p ./data
scp -r adlm:/vol/chameleon/projects/adni/Tabular_Info_PET.csv ./data/
scp -r adlm:/vol/chameleon/projects/adni/ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv ./data/