#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=1:0:0    # 24 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=1     # request 1 GPU
##$ -l cluster=andrena # use the Andrena nodes

# test blur:0.5,1,5; test: use_origin_img; dilation=20,40,100

module load python
source ~/pytorchenv1/bin/activate

cd /data/DERI-Gong/jl010/Seg/zero-shot-hard-sample-segemetation

# use_origin_neg_points 
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation --use_origin_neg_points >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_originImg_OriNegPt.log
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation --use_origin_neg_points >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_OriNegPt.log

python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation --use_origin_neg_points  >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_originImg_OriNegPt.log

# python demo_dataset_.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation --use_origin_neg_points --test_vis_dir='PostBoxMaskWoPt1.1' >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_originImg_OriNegPt_PostBoxMaskWoPt1.1.log



## use_origin_img
# dilation=20
# python demo_dataset.py --recursive_blur_gauSigma=0.5 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma0.5_inflateK20_mask_originImg.log
# python demo_dataset.py --recursive_blur_gauSigma=1 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma1_inflateK20_mask_originImg.log
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_originImg.log

# dilation=40
# python demo_dataset.py --recursive_blur_gauSigma=0.5 --dilation_k=100 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma0.5_inflateK100_mask_originImg.log
# python demo_dataset.py --recursive_blur_gauSigma=1 --dilation_k=100 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma1_inflateK100_mask_originImg.log
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=100 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK100_mask_originImg.log



# ## wo use_origin_img
# # dilation=20
# python demo_dataset.py --recursive_blur_gauSigma=0.5 --dilation_k=20 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma0.5_inflateK20_mask.log
# python demo_dataset.py --recursive_blur_gauSigma=1 --dilation_k=20 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma1_inflateK20_mask.log
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=20 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask.log

# # dilation=40
# python demo_dataset.py --recursive_blur_gauSigma=0.5 --dilation_k=100 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma0.5_inflateK100_mask.log
# python demo_dataset.py --recursive_blur_gauSigma=1 --dilation_k=100 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma1_inflateK100_mask.log
# python demo_dataset.py --recursive_blur_gauSigma=5 --dilation_k=100 --use_fuse_mask_hm     --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK100_mask.log




# python demo_dataset.py --recursive_blur_gauSigma=0.5 --dilation_k=100 --use_fuse_mask_hm --use_origin_img      --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6  --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma0.5_inflateK100_mask_originImg.log
