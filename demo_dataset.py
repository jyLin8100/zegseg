

from pickle import FALSE
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import argparse
import yaml
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_model import get_text_from_img, get_mask, get_fused_mask, printd, reset_params, get_dir_from_args
import os

def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)


## configs
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/mydemo.yaml')
parser.add_argument('--multi_mask_fusion', type=bool, default=False, help='fuse multiple masks') 
parser.add_argument('--multi_mask_fusion_strategy', type=str, default='avg', help='fuse multiple masks')  # avg, entropy, entropy2
parser.add_argument('--cache_blip_filename', default='COD_GT_woPos') # COD, COD_woPos, COD_GT, COD_GT_woPos, COD_BLIP_GT_woPos
parser.add_argument('--clip_model', type=str, default='CS-ViT-B/16', help='model for clip surgery') 
parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='') 
parser.add_argument('--sam_model_type', type=str, default='vit_h', help='') 

parser.add_argument('--down_sample', type=float, default=2, help='down sample to generate points from CLIP surgery output') 
parser.add_argument('--attn_thr', type=float, default=0.95, help='threshold for CLIP Surgery to get points from attention map') 
parser.add_argument('--pt_topk', type=int, default=-1, help='for CLIP Surgery to get points from attention map, use points of top k highest socre as positive sampling points') 
parser.add_argument('--recursive', type=int, default=0, help='recursive times to use CLIP surgery, to get the point') 
parser.add_argument('--recursive_coef', type=float, default=0.3, help='recursive coefficient to use CLIP surgery, to get the point') 
parser.add_argument('--recursive_blur_gauSigma', type=float, default=1, help='sigma for guassian blur') 
parser.add_argument('--recursive_input_strategy', type=float, default=1, help='strategy for refining the img input of clip surgery: accum, last') 

parser.add_argument('--use_origin_img', action='store_true') 
parser.add_argument('--use_dilation', action='store_true') 
parser.add_argument('--dilation_k', type=int, default=40, help='') 
parser.add_argument('--use_blur', action='store_true') 
parser.add_argument('--use_fuse_mask_hm', action='store_true') 
parser.add_argument('--use_origin_neg_points', action='store_true') 
parser.add_argument('--add_origin_neg_points', action='store_true') 
parser.add_argument('--post_mode', type=str, default='', help='') 
parser.add_argument('--rdd_str', type=str, default='', help='text for redundant features as input of clip surgery') 
parser.add_argument('--clip_use_neg_text', type=bool, default=False, help='negative text input for clip surgery') 
parser.add_argument('--clip_neg_text_attn_thr', type=float, default=0.8, help='negative threshold for clip surgery') 
parser.add_argument('--clip_attn_qkv_strategy', type=str, default='vv', help='qkv attention strategy for clip surgery')  # vv(original), kk

parser.add_argument('--test', action='store_true')  # store output in output_img_test
parser.add_argument('--test_vis_dir', type=str, default='', )  # store output in output_img_test

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
spec = config['test_dataset']
dataset_name = spec['dataset']['args']['root_path_1'].split('/')[2]
if args.cache_blip_filename is not None:
    cache_blip_filename = f'blip_cache/{args.cache_blip_filename}'
else:
    cache_blip_filename = f'blip_cache/{dataset_name}'
blip_text_l = []
if os.path.exists(cache_blip_filename):
    cache_blip_file = open(cache_blip_filename, "r")
    for text in cache_blip_file:
        blip_text_l.append(text[:-1].split(','))
    printd(f"loading BLIP text output from file: {cache_blip_filename}, length:{len(blip_text_l)}")

parent_dir = f'output_img/{cache_blip_filename}/'
if args.test:   parent_dir = f'output_img_test/{cache_blip_filename}_{args.test_vis_dir}/'
save_path_dir = get_dir_from_args(args, config, parent_dir)


mkdir(save_path_dir)
printd(f'save_path_dir: {save_path_dir}')

## get data
printd(f'loading dataset...')
dataset = datasets.make(spec['dataset'])
dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
loader = DataLoader(dataset, batch_size=spec['batch_size'],
                    num_workers=8)
paths_img = dataset.dataset.paths_img
data_len = len(paths_img)
printd(f"dataset size:\t {len(paths_img)}")
if len(blip_text_l)>0:
    assert data_len==len(blip_text_l)

## load model  
from segment_anything import sam_model_registry, SamPredictor
from clip.clip_surgery_model import CLIPSurgery
import clip
sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
clip_params={ 'attn_qkv_strategy':args.clip_attn_qkv_strategy}
clip_model, _ = clip.load(args.clip_model, device=device, params=clip_params)
clip_model.eval()
use_cache_blip = True
if len(blip_text_l)==0:
    use_cache_blip = False
    from lavis.models import load_model_and_preprocess
    # blip_model_type="pretrain_opt2.7b"
    blip_model_type="pretrain_opt6.7b" 
    printd(f'loading BLIP ({blip_model_type})...')
    BLIP_model, BLIP_vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type=blip_model_type, is_eval=True, device=device)
    BLIP_dict = {"demo_data/9.jpg": 'lizard in the middle',}

## metrics
import utils
metric_fn = utils.calc_cod
metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
val_metric1 = [utils.Averager() for i in range(args.recursive+1)]
val_metric2 = [utils.Averager() for i in range(args.recursive+1)]
val_metric3 = [utils.Averager() for i in range(args.recursive+1)]
val_metric4 = [utils.Averager() for i in range(args.recursive+1)]


## run model
for s_i, img_path, pairs in zip(range(data_len), paths_img, loader):

    pil_img = Image.open(img_path).convert("RGB")
    if use_cache_blip:
        text = blip_text_l[s_i] 
    else:
        text = get_text_from_img(img_path, pil_img, BLIP_dict, BLIP_model, BLIP_vis_processors, device)
    

    if args.multi_mask_fusion:
        mask, mask_logit, points, labels, num, vis_dict = get_fused_mask(pil_img, text, sam_predictor, clip_model, args, device, config)
    else:
        mask, mask_logit, _, points, labels, num, vis_dict = get_mask(pil_img, text, sam_predictor, clip_model, args, device)
        vis_map_img = vis_dict['vis_map_img']
        vis_input_img = vis_dict['vis_input_img']
        vis_radius = vis_dict['vis_radius']
        vis_mask_l = vis_dict['vis_mask_l']
        vis_mask_logit_l = vis_dict['vis_mask_logit_l']
        vis_radius_l = vis_dict['vis_radius_l']
        points_l = vis_dict['points_l']
        labels_l = vis_dict['labels_l']
        num_l = vis_dict['num_l']
        vis_clip_sm_img = vis_dict['vis_clip_sm_img']

    ## metric
    # align size of GT mask first
    # tensor_img = pairs['inp']
    tensor_gt = pairs['gt']
    inp_size = 1024
    mask_transform = transforms.Compose([
                    transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])
    printd(f'{s_i}\t img_path:{img_path}\t text:{text}\t  ')
    for i in range(args.recursive+1):
        vis_tensor = Image.fromarray(vis_mask_l[i])
        vis_tensor = mask_transform(vis_tensor)[0].view(1, 1, inp_size, inp_size)
        result1, result2, result3, result4 = metric_fn(vis_tensor, tensor_gt)
        
        print(f'{result1:.3f} {result2:.3f} {result3:.3f} {result4:.3f} num of points: {num_l[i]}')
        val_metric1[i].add(result1.item(), tensor_gt.shape[0])
        val_metric2[i].add(result2.item(), tensor_gt.shape[0])
        val_metric3[i].add(result3.item(), tensor_gt.shape[0])
        val_metric4[i].add(result4.item(), tensor_gt.shape[0])


    ## visualization
    if s_i%1==0 and s_i<10:
        img_name = img_path.split('/')[-1][:-4]
        vis_pt_l = [np.expand_dims(255*vis_mask_l[i], axis=2).repeat(3, axis=2) for i in range(len(vis_mask_l))]
        if not args.multi_mask_fusion:
            for j in range(len(points_l)):
                for i, [x, y] in enumerate(points_l[j]):
                    if labels_l[j][i] == 0:
                        clr = (0, 102, 255)
                    elif labels_l[j][i] == 1:
                        clr = (255, 102, 51)
                    else:
                        clr = (0, 255, 102)
                    cv2.circle(vis_pt_l[j], (x, y), vis_radius_l[j][i], clr, vis_radius_l[j][i])
                    cv2.circle(vis_input_img[j], (x, y), vis_radius_l[j][i], clr, vis_radius_l[j][i])
        
            for i in range(len(vis_map_img)):
                plt.imsave(save_path_dir + img_name + f'_iptImg{i}.jpg', vis_input_img[i])
                plt.imsave(save_path_dir + img_name + f'_meanSm{i}.jpg', vis_clip_sm_img[i])
                plt.imsave(save_path_dir + img_name + f'_maskLog{i}.jpg', vis_mask_logit_l[i], cmap='gray')
                plt.imsave(save_path_dir + img_name + f'_fuseSm{i}.jpg', vis_map_img[i])
                plt.imsave(save_path_dir + img_name + f'_sam_pt{i}.jpg', vis_pt_l[i])
                if len(vis_dict['vis_mask0_l'])>i:
                    plt.imsave(save_path_dir + img_name + f'_mask0_{i}.jpg', vis_dict['vis_mask0_l'][i], cmap='gray')
        
        save_path_sam_pt = save_path_dir + img_name + f"_sam_pt.jpg"
        save_path_sam_pt_logit = save_path_dir + img_name + f"_sam_pt_logit.jpg"
        plt.imsave(save_path_sam_pt, vis_pt_l[-1])
        plt.imsave(save_path_sam_pt_logit, mask_logit, cmap='gray')

        # save_path_sam_pt_logit_img = save_path_dir + img_name + f"_sam_pt_logit_img.jpg"
        # logit_img = mask_logit/255*vis_input_img[0]
        # plt.imsave(save_path_sam_pt_logit_img, logit_img.astype('uint8'))
        # save_path_sam_pt_img = save_path_dir + img_name + f"_sam_pt_img.jpg"
        # mask_img = vis_pt/255*vis_input_img[0]
        # plt.imsave(save_path_sam_pt_img, mask_img.astype('uint8'))
        
        # save_path_sam = save_path_dir + img_name + f"_sam.jpg"
        # save_path_gt = save_path_dir + img_name + f"_gt.jpg"
        # plt.imsave(save_path_sam, vis_tensor.view(1024,1024).numpy(), cmap='gray')
        # plt.imsave(save_path_gt, tensor_gt.view(1024,1024).numpy(), cmap='gray')
for i in range(args.recursive+1):
    print(val_metric1[i].item(),                
          val_metric2[i].item(),
          val_metric3[i].item(),
          val_metric4[i].item(),)
 