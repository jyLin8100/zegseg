#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=50:0:0    # 24 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=1     # request 1 GPU
##$ -l cluster=andrena # use the Andrena nodes

# note: try modifying inputs for clip, using blur, ema, fuse mask, use original neg points, post refienment for SAM (6-28) 

module load python
source ~/pytorchenv1/bin/activate

cd /data/DERI-Gong/jl010/Seg/zero-shot-hard-sample-segemetation

# python demo_dataset.py 
#  --use_fuse_mask_hm 
#  --use_origin_img
#  --use_blur --recursive_blur_gauSigma=5 
#  --use_dilation  --dilation_k=20
#  --use_origin_neg_points   --recursive_coef=1  \
# --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 6 --clip_attn_qkv_strategy='kk'   >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur6_coef_kk_blur_sigma5_inflateK20_mask_originImg_OriNegPt.log

# baseline
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk.log  


## 5.post refine for SAM
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostBoxWoPt' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostBoxWoPt.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostBoxMaskWoPt1.1' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostBoxMaskWoPt1.1.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostLogit' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostLogit.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='LogitSAMInput' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postLogitSAMInput.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostBox' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostBox.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostBoxWoPt1.1' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostBoxWoPt1.1.log 
python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='PostBoxMaskWoPtInflate1.1' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postPostBoxMaskWoPtInflate1.1.log 



# ---------------------------------------------------------------------------------------------

## 1. --use_fuse_mask_hm  --use_origin_img
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5   --use_fuse_mask_hm >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_mask.log  
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5   --use_fuse_mask_hm --use_origin_img >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_mask_originImg.log  
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5   --use_origin_img >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_originImg.log  

## 2. use blur1 
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_blur1_sigma5.log  # baseline
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5  --use_fuse_mask_hm >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_blur1_sigma5_mask.log  # baseline

# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_coef=0.5  --recursive_blur_gauSigma=5  >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5.log  # 1
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_coef=0.7  --recursive_blur_gauSigma=5  >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.7_kk_blur1_sigma5.log  # 1
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=1 >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_blur1_sigma1.log  # baseline

## 3. clipInputEMA darken ( 感觉更合理，不会马上变得更黑；且错误的地方可纠正)
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_clipInputEMA.log  # 1
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --recursive_coef=0.5 --clipInputEMA >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.5_kk_clipInputEMA.log  # 1
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --recursive_coef=0.7 --clipInputEMA >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.7_kk_clipInputEMA.log  # 1
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=10 --recursive_coef=0.1 --clipInputEMA >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur10_0.1_kk_clipInputEMA.log  # 

## 4. use_origin_neg_points
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --use_origin_neg_points >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_OriNegPt.log  
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --add_origin_neg_points >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_addOriNegPt.log  



