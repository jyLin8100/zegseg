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

cd /data/DERI-Gong/jl010/Seg/zero-shot-hard-sample-segemetation

# python demo_dataset_.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short.log 
# python demo_dataset_2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3shortWoIC' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3shortWoIC.log 
# python demo_dataset_3.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attri' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attri.log 
# python demo_dataset_4.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short1long' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short1long.log 
# python demo_dataset_5.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short2IC' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short2IC.log 

# python demo_dataset_6.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriCreature' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriCreature.log 
# python demo_dataset_7.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriTheCreature' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriTheCreature.log 
# python demo_dataset_8.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriTheCreatureBriefly' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriTheCreatureBriefly.log 
# python demo_dataset_9.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriTheCreatureShortly' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriTheCreatureShortly.log 
# python demo_dataset_10.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriThe' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriThe.log 
python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attriThe' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attriThe.log 

# python demo_dataset_t.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short.log 
# python demo_dataset_2t.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3shortWoIC' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3shortWoIC.log 
# python demo_dataset_3t.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3attri' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3attri.log 
# python demo_dataset_4t.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short1long' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short1long.log 
# python demo_dataset_5t.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir '3short2IC' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_3short2IC.log 



### use heatmap to get the most possible mask
## python demo_dataset_.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir 'fuse1'  >> output_log2_test/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_fuse1.log 
## python demo_dataset_2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --test --test_vis_dir 'fuse1mask'  >> output_log2_test/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_fuse1mask.log 

## python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'getIndex'  >> output_log2_test/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
## python demo_dataset_bak.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'getIndex'  >> output_log2_test/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
## python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'woOneWordLimit'  >> output_log2_test/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_woOneWordLimit.log 
## python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'woOneWordLimit'  >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_woOneWordLimit.log 
## python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'concealed'  >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealed.log 
## python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --test --test_vis_dir 'camouflage'  >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflage.log 

# --------------------------------------------------------------------------
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hidden.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOne.log 

# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealed.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedOne.log 

# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflage.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --prompt_q='Name of camouflage animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_camouflage.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageOne.log 


# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenFS.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneFS.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedFS.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedOneFS.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageFS.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageOneFS.log 


# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden object' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenObject.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden object in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneObject.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden creature' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenCreature.log 
# python demo_dataset_analysis1.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden creature in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneCreature.log 

# --------------------------------------------------------------------------


# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hidden.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOne.log 

# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealed.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedOne.log 

# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflage.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --prompt_q='Name of camouflage animal' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_camouflage.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageOne.log 


# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenFS.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneFS.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedFS.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of concealed animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_concealedOneFS.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageFS.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of camouflage animal in one word.' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_camouflageOneFS.log 


# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden object' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenObject.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden object in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneObject.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden creature' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenCreature.log 
# python demo_dataset_analysis2.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --prompt_q='Name of hidden creature in one word' >> output_log_analysis1/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_hiddenOneCreature.log 

# --------------------------------------------------------------------------





# --------------------------------------------------------------------------

  # baseline
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk.log 

# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5

## 4. fuse mask
# python demo_dataset.py    --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA   --post_mode='MaxIOUBoxSAMInput'  --use_fuse_mask_hm  >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_mask.log 
# python demo_dataset.py    --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'  --use_fuse_mask_hm  >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput_mask.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA   --post_mode='MaxIOUBoxSAMInput'  --use_fuse_mask_hm >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput_mask.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'  --use_fuse_mask_hm>> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput_mask.log 


## 1. mulhead & MaxIOUBoxSAMInput
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --multi_head >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_mulHead.log  #
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --multi_head --post_mode='MaxIOUBoxSAMInput' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_mulHead_postMaxIOUBoxSAMInput.log  #
# python demo_dataset_.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --post_mode='MaxIOUBoxSAMInput' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_postMaxIOUBoxSAMInput.log  #


## 2.COD_GT_woPos_mulWords , COD_woPos
# python demo_dataset.py --cache_blip_filename COD_GT_woPos_mulWords --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' >> output_log2/COD_GT_woPos_mulWords_thr9e-1_s05_kk.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos_mulWords --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 >> output_log2/COD_GT_woPos_mulWords_thr9e-1_s05_rcur5_kk.log  #2993890
# python demo_dataset.py --cache_blip_filename COD_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 >> output_log2/COD_woPos_thr9e-1_s05_rcur5_kk.log 

## 3. iterative update blip (6-28)
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_MaxIOUBoxSAMInput.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA   --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5 >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5.log 
# python demo_dataset.py   --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 

# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  >> output_log2/BlipText_thr9e-1_s05_rcur5_kk.log 
# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipText_thr9e-1_s05_rcur5_kk_MaxIOUBoxSAMInput.log 
# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA >> output_log2/BlipText_thr9e-1_s05_rcur5_kk_clipInputEMA.log 
# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA   --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipText_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5 >> output_log2/BlipText_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5.log 
# python demo_dataset.py   --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipText_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 

# python demo_dataset.py    --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA   --post_mode='MaxIOUBoxSAMInput' >> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py    --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/COD_GT_woPos_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 

## 调参 recursive_coef, down_sample, recursive, post_mode
# baseline: python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --recursive_coef=0.4 >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.4_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --recursive_coef=0.2 >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.2_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' --recursive_coef=0.3 >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.4_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=1 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  >> output_log2/BlipTextUpdate_thr9e-1_s1_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.25 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  >> output_log2/BlipTextUpdate_thr9e-1_s025_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.95 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipText_thr95e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.85 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipText_thr85e-1_s05_rcur5_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='PostBox' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_PostBox.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5 --clipInputEMA --post_mode='PostLogit' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_kk_clipInputEMA_PostLogit.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=10 --clipInputEMA --post_mode='MaxIOUBoxSAMInput' >> output_log2/BlipTextUpdate_thr9e-1_s05_rcur10_kk_clipInputEMA_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=10 --clipInputEMA --post_mode='MaxIOUBoxSAMInput'  --recursive_coef=0.2 >> output_log2/BlipTextUpdate_thr9e-1_s05_0.2_rcur10_kk_clipInputEMA_MaxIOUBoxSAMInput.log 

# baseline: python demo_dataset.py --use_cache_text '' --update_text --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipTextUpdate_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=3 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipText_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=7 --recursive_coef=0.5  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipText_thr9e-1_s05_rcur5_0.5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --recursive_coef=0.7  --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipText_thr9e-1_s05_rcur5_0.7_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 
# python demo_dataset.py --use_cache_text ''  --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=5  --use_blur1 --recursive_blur_gauSigma=5 --post_mode='MaxIOUBoxSAMInput'>> output_log2/BlipText_thr9e-1_s05_rcur5_kk_blur1_sigma5_MaxIOUBoxSAMInput.log 




# -------------------------------------------------------------------------------
# python demo.py
# python demo_dataset.py --cache_blip_filename COD_woPos >> COD_woPos.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos >> output_log/COD_GT_woPos.log

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
# kk
# python demo_dataset.py --multi_mask_fusion=True --config config/s1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.9.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s2_1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9_kk.log
# echo cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9_kk_entrp.log; python demo_dataset.py --multi_mask_fusion=True --config config/s2_1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' --multi_mask_fusion_strategy='entropy' >> output_log/cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9_kk_entrp.log

## fuse threshold
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_0.95.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95.log
# kk
# echo cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95_kk.log; python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95_kk.log
# echo cfg_COD_GT_woPos_s0.5_thr0.8_0.85_0.9_0.95_kk.log; python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.8_0.85_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.85_0.9_0.95_kk.log
# echo cfg_COD_GT_woPos_s0.5_thr0.8_0.85_0.9_0.95_kk_entrp.log; python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='entropy' --config config/s0.5_thr0.8_0.85_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.85_0.9_0.95_kk_entrp.log

## fuse recursive
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4.log; python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3_4.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4.log
# kk
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_kk.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1.yaml  --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk'>> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_kk.log
# python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk; python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4_kk; python demo_dataset.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3_4.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4_kk.log

## fusing method: multi_mask_fusion_strategy (avg: refined mask_logits; entropy, entropy2) -> still use avg
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp; python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='entropy' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp_log2.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp; python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='entropy' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp2; python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='entropy2' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_entrp2.log
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4_kk_entrp.log; python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='entropy' --config config/s0.5_thr0.9_rcur0_1_2_3_4.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_4_kk_entrp.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1.log
# python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='hmMseTop1'  --config config/s2_1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9_kk_hmMseTop1.log
# python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='hmMseTop1'  --config config/s2_1_0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s2_1_0.5_thr0.9_rcur0_1_2_kk_hmMseTop1.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.5' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr0.5.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.8' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr0.8.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.9' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr0.9.log

# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.5' --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk_hmMseTop1thr0.5.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.8' --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk_hmMseTop1thr0.8.log
# python demo_dataset.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.9' --config config/s0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_kk_hmMseTop1thr0.9.log

## fuse recursive & scale
# python demo_dataset_bak.py --multi_mask_fusion=True --config config/s2_1_0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s2_1_0.5_thr0.9_rcur0_1_2_kk.log
# python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='hmMseTop1thr' --config config/s2_1_0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s2_1_0.5_thr0.9_rcur0_1_2_kk_hmMseTop1thr.log
# python demo_dataset.py --multi_mask_fusion=True --multi_mask_fusion_strategy='hmMseTop1thr0.5' --config config/s2_1_0.5_thr0.9_rcur0_1_2.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s2_1_0.5_thr0.9_rcur0_1_2_kk_hmMseTop1thr0.5.log



# ---------------to visualize bad sample -------------------#

# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_kk.log  
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=1 >> output_log_analysis/COD_GT_woPos_thr8e-1_s05_rcur1_kk.log  
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=1 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur1_kk.log  
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=2 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur2_kk.log  
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur3_kk.log  
# echo cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk; python demo_dataset_analysis.py --multi_mask_fusion=True --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk.log
# python demo_dataset_analysis.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1.log
# python demo_dataset_analysis.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr.log
# python demo_dataset_analysis.py --multi_mask_fusion=True  --multi_mask_fusion_strategy='hmMseTop1thr0.5' --config config/s0.5_thr0.9_rcur0_1_2_3.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s0.5_thr0.9_rcur0_1_2_3_kk_hmMseTop1thr0.5.log




# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9  --clip_attn_qkv_strategy='kk' >> output_log_analysis/COD_GT_woPos_thr9e-1_s2_kk.log 
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=1 --clip_attn_qkv_strategy='kk' >> output_log_analysis/COD_GT_woPos_thr9e-1_s1_kk.log 
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5  --clip_attn_qkv_strategy='kk'>> output_log_analysis/COD_GT_woPos_thr9e-1_s05_kk.log  # best
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.25  --clip_attn_qkv_strategy='kk'>> output_log_analysis/COD_GT_woPos_thr9e-1_s025_kk.log 
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.125  --clip_attn_qkv_strategy='kk'>> output_log_analysis/COD_GT_woPos_thr9e-1_s0125_kk.log 
# python demo_dataset_analysis.py --multi_mask_fusion=True --config config/s2_1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos  --clip_attn_qkv_strategy='kk' >> output_log_analysis/cfg_COD_GT_woPos_s2_1_0.5_0.25_thr0.9_kk.log


# ----------- test -------------
## test
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2
# echo COD_GT_woPos_thr8e-1_s1_neg0.95; python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --clip_attn_qkv_strategy='kk' >> out.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.5


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3 --clip_attn_qkv_strategy='kk' --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_kk_blur.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=0.5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_kk_blur_sigma0.5.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_kk_blur_sigma5.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=1 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma1.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=0.5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma0.5.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 2  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=0.5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur2_coef_kk_blur_sigma0.5.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 1  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=0.5 --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur1_coef_kk_blur_sigma0.5.log


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflate.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=1 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma1_inflate.log


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK20.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=1 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma1_inflateK20.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK10.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=1 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma1_inflateK10.log


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK20.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK20_mask.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK40_mask.log
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK40_mask_originImg_.log


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk' --recursive_blur_gauSigma=5 --test >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK40_mask_originImg_.log

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3  --recursive_coef=1 --clip_attn_qkv_strategy='kk'  --test  --use_blur --use_dilation \
# --recursive_blur_gauSigma=5 \
# --dilation_k=20 \
# --use_fuse_mask_hm \
# --use_origin_img \
#  >> output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_coef_kk_blur_sigma5_inflateK20_mask_originImg.log





# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 3 --clip_attn_qkv_strategy='kk' --test >>output_log_test/COD_GT_woPos_thr9e-1_s05_rcur3_kk_blur.log
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --recursive 1  --clip_attn_qkv_strategy='kk' --test 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --test
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --test 


# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log/COD_GT_woPos_thr9e-1_s05_rcur3_kk_add0.1.log 
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur3_kk_add0.1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log/COD_GT_woPos_thr9e-1_s05_rcur3_kk.log 
# python demo_dataset_analysis.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur3_kk_add0.2.log  
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk' --recursive=3 >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_rcur3_kk_woNeg.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 --clip_attn_qkv_strategy='kk'  >> output_log_analysis/COD_GT_woPos_thr9e-1_s05_kk_woNeg.log 







###############################################
## ------------------ bak ------------------ ##
###############################################

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

# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=1 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur1_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur2_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=3 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur3_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=4 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur4_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=5 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur5_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=10 --recursive_coef=0.1 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur10_coef1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=1 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur1_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=2 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur2_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=3 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur3_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=4 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur4_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=5 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur5_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=10 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur10_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=15 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur15_coef05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --recursive=20 --recursive_coef=0.05 >> output_log/COD_GT_woPos_thr8e-1_s1_rcur20_coef05.log 

## background
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=1 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=2 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=1 --rdd_str='background' --recursive=3 >> output_log/COD_GT_woPos_thr8e-1_s1_rddbackground_rcur3.log 

## downsample
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.5 >> output_log/COD_GT_woPos_thr8e-1_s05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.25 >> output_log/COD_GT_woPos_thr8e-1_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.8 --down_sample=0.125 >> output_log/COD_GT_woPos_thr8e-1_s0125.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9  >> output_log/COD_GT_woPos_thr9e-1_s2.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9  --down_sample=1 >> output_log/COD_GT_woPos_thr9e-1_s1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.5 >> output_log/COD_GT_woPos_thr9e-1_s05.log  # best
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.25 >> output_log/COD_GT_woPos_thr9e-1_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.9 --down_sample=0.125 >> output_log/COD_GT_woPos_thr9e-1_s0125.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=1 >> output_log/COD_GT_woPos_thr95e-2_s1.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.5 >> output_log/COD_GT_woPos_thr95e-2_s05.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.25 >> output_log/COD_GT_woPos_thr95e-2_s025.log 
# python demo_dataset.py --cache_blip_filename COD_GT_woPos --attn_thr 0.95 --down_sample=0.125 >> output_log/COD_GT_woPos_thr95e-2_s0125.log 



# ----------- multiple mask -------------
    # bak: script for  demo_dataset_fuse.py
    ## fuse scale
    # python demo_dataset_fuse.py --config config/s1_0.5_0.25_thr0.9.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.9.log
    # python demo_dataset_fuse.py --config config/s1_0.5_0.25_thr0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s1_0.5_0.25_thr0.95.log

    ## fuse threshold
    # python demo_dataset_fuse.py --config config/s0.5_thr0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.9_0.95.log
    # python demo_dataset_fuse.py --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95.log
    # python demo_dataset_fuse.py --config config/s0.5_thr0.8_0.9_0.95.yaml --cache_blip_filename COD_GT_woPos --clip_attn_qkv_strategy='kk' >> output_log/cfg_COD_GT_woPos_s0.5_thr0.8_0.9_0.95_kk.log

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

