#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=1:0:0    # 24 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=1     # request 1 GPU
##$ -l cluster=andrena # use the Andrena nodes

module load python
source ~/pytorchenv1/bin/activate
# yes y| pip install  git+https://github.com/facebookresearch/segment-anything.git
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# yes y| pip install salesforce-lavis
# yes y| pip install scikit-learn
# yes y| pip install tensorboardX

cd /data/DERI-Gong/jl010/Seg/zero-shot-hard-sample-segemetation
# python demo.py
# python demo_dataset.py --cache_blip_filename COD_woPos >> COD_woPos.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos >> output_log/COD_GT_woPos.log


## topk
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0 --pt_topk 1 >> output_log/COD_GT_woPos_top1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0 --pt_topk 3 >> output_log/COD_GT_woPos_top3.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --pt_topk 1 >> output_log/COD_GT_woPos_thr8e-1_top1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --pt_topk 3 >> output_log/COD_GT_woPos_thr8e-1_top3.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8  >> output_log/COD_GT_woPos_thr8e-1.log 

## downsample
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 >> output_log/COD_GT_woPos_thr8e-1_s1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=2 >> output_log/COD_GT_woPos_thr8e-1_s2.log 

## recurvise
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=0 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur0.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=3 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur3.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=1 --recursive_coef=0.2 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur1_coef2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2 --recursive_coef=0.2 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur2_coef2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=3 --recursive_coef=0.2 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur3_coef2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=1 --recursive_coef=0.4 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur1_coef4.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2 --recursive_coef=0.4 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur2_coef4.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=3 --recursive_coef=0.4 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur3_coef4.log 

## background
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=1 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=2 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=3 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur3.log 

## downsample
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.5 >> output_log/COD_GT_woPos_thr8e-1_s05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.25 >> output_log/COD_GT_woPos_thr8e-1_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.125 >> output_log/COD_GT_woPos_thr8e-1_s0125.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9  >> output_log/COD_GT_woPos_thr9e-1_s1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 >> output_log/COD_GT_woPos_thr9e-1_s05.log  # best
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.25 >> output_log/COD_GT_woPos_thr9e-1_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.125 >> output_log/COD_GT_woPos_thr9e-1_s0125.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=1 >> output_log/COD_GT_woPos_thr95e-2_s1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.5 >> output_log/COD_GT_woPos_thr95e-2_s05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.25 >> output_log/COD_GT_woPos_thr95e-2_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.125 >> output_log/COD_GT_woPos_thr95e-2_s0125.log 

## neg
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_use_neg_text=True --clip_neg_text_attn_thr=0.8 >> output_log/COD_GT_woPos_thr8e-1_s1_neg.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_use_neg_text=True --clip_neg_text_attn_thr=0.9 >> output_log/COD_GT_woPos_thr8e-1_s1_neg0.9.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_use_neg_text=True --clip_neg_text_attn_thr=0.95 >> output_log/COD_GT_woPos_thr8e-1_s1_neg0.95.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_use_neg_text=True >> output_log/COD_GT_woPos_thr8e-1_s1_neg_class2.log 


# ----------- multiple mask -------------

## fuse scale
# python demo_dataset.py --multi_mask_fusion=True --config config/s1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.9.log
# echo cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9.log; python demo_dataset.py --multi_mask_fusion=True --config config/s2_1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s1_0.5_0.25_thr0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.95.log

## fuse threshold
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_0.95.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95_kk.log

## fuse recursive
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3.log
# kk
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_kk.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_kk.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk.log






### fuse scaledemo_dataset_fuse.py
# bak: script for 
## fuse scale
# python demo_dataset_fuse.py --config config/s1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.9.log
# python demo_dataset_fuse.py --config config/s1_0.5_0.25_thr0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.95.log

## fuse threshold
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_0.95.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95.log
python demo_dataset_fuse.py --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95_kk.log

## fuse recursive
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3.log
# kk
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_kk.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_kk.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk.log
# python demo_dataset_fuse.py --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk.log


# ----------- test -------------
## test
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2
# echo COD_GT_woPos_thr8e-1_s1_neg0.95; python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_attn_qkv_strategy='kk' >> out.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.5