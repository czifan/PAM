#!/bin/bash

datasets=("Adrenal-ACC-Ki67-Seg" "CHAOS-CT" "COVID-19 Seg. Challenge" "KiPA" "KiTS" "AbdomenCT-1K" "INSTANCE" "LNQ2023" "Lymph Nodes" "NSCLC Pleural Effusion" "QUBIQ-CT" "Task03_Liver" "Task06_Lung" "Task07_Pancreas" "Task08_HepaticVessel" "Task09_Spleen" "Task10_Colon" "WORD")

for dataset in "${datasets[@]}"
do
    python infer.py --box2seg_model "None" --infer_dir "infer" --dataset $dataset
done