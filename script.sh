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
# # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# yes y| pip install salesforce-lavis

cd /data/DERI-Gong/jl010/Seg/zero-shot-hard-sample-segemetation
python demo.py



