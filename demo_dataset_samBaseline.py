

from pickle import FALSE
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import argparse
import yaml
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import os
BICUBIC = InterpolationMode.BICUBIC

def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)


## configs
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/mydemo.yaml')
parser.add_argument('--cache_blip_filename', default='COD_GT_woPos') # COD, COD_woPos, COD_GT, COD_GT_woPos, COD_BLIP_GT_woPos
parser.add_argument('--down_sample', type=int, default=2, help='down sample to generate points from CLIP surgery output') 
parser.add_argument('--attn_thr', type=float, default=0.95, help='threshold for CLIP Surgery to get points from attention map') 
parser.add_argument('--pt_topk', type=int, default=-1, help='for CLIP Surgery to get points from attention map, use points of top k highest socre as positive sampling points') 
parser.add_argument('--recursive', type=int, default=0, help='recursive times to use CLIP surgery, to get the point') 
parser.add_argument('--recursive_coef', type=float, default=0.3, help='recursive coefficient to use CLIP surgery, to get the point') 

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(f'args:\n{args}\n')
spec = config['test_dataset']
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = spec['dataset']['args']['root_path_1'].split('/')[2]

save_dir_name = f'{dataset_name}'

save_path_dir = f'output_img_samBaseline/{save_dir_name}/'
mkdir(save_path_dir)
print(f'save_path_dir: {save_path_dir}')

## get data
print(f'loading dataset...')
dataset = datasets.make(spec['dataset'])
dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
loader = DataLoader(dataset, batch_size=spec['batch_size'],
                    num_workers=8)
paths_img = dataset.dataset.paths_img
data_len = len(paths_img)
print("dataset size:", len(paths_img))

## load model
# clip
import clip
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()
# Init SAM
from segment_anything import sam_model_registry, SamPredictor
# from sam_utils import show_mask, show_points, show_box
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

## metrics
import utils
metric_fn = utils.calc_cod
metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
val_metric1 = utils.Averager()
val_metric2 = utils.Averager()
val_metric3 = utils.Averager()
val_metric4 = utils.Averager()

for s_i, img_path, pairs in zip(range(data_len),paths_img, loader):
    tensor_img = pairs['inp']
    tensor_gt = pairs['gt']
    pil_img = Image.open(img_path).convert("RGB")
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    predictor.set_image(np.array(pil_img))

    with torch.no_grad():
        # Inference SAM with points from CLIP Surgery
        # masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True)
        h, w, _ = np.array(pil_img).shape
        input_box = np.array([0, 0, w-1, h-1])
        # masks, scores, logits = predictor.predict(point_coords=None,
        #                                             point_labels=None,
        #                                             box=input_box[None, :], 
        #                                             multimask_output=True)
        # mask = masks[np.argmax(scores)]
        masks, scores, logits = predictor.predict(point_coords=None,
                                                    point_labels=None,
                                                    box=input_box[None, :], 
                                                    multimask_output=False)
        mask = masks[0]
        mask = mask.astype('uint8')
        # Visualize the results
        vis = cv2_img[...,0].copy()
        vis[mask > 0] = np.array(255, dtype=np.uint8) 
        vis[mask == 0] = np.array(0, dtype=np.uint8)
        vis_pt = np.expand_dims(vis, axis=2).repeat(3, axis=2)
        cv2.rectangle(vis_pt, (0,0), (w-1,h-1), (0, 102, 255), 3)


        # align size of GT mask
        inp_size = 1024
        mask_transform = transforms.Compose([
                        transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                        transforms.ToTensor(),
                    ])
        vis_tensor = Image.fromarray(vis)
        vis_tensor = mask_transform(vis_tensor)[0].view(1, 1, inp_size, inp_size)


        ## metric
        result1, result2, result3, result4 = metric_fn(vis_tensor, tensor_gt)
        print(f'img_path:{img_path}\t {result1:.3f} {result2:.3f} {result3:.3f} {result4:.3f}')
        val_metric1.add(result1.item(), tensor_gt.shape[0])
        val_metric2.add(result2.item(), tensor_gt.shape[0])
        val_metric3.add(result3.item(), tensor_gt.shape[0])
        val_metric4.add(result4.item(), tensor_gt.shape[0])




        ## visualization
        if s_i%1==0 and s_i<10:
            img_name = img_path.split('/')[-1][:-4]
            save_path_sam_pt = save_path_dir + img_name + f"_sam_pt.jpg"
            plt.imsave(save_path_sam_pt, vis_pt)

            # save_path_sam = save_path_dir + img_name + f"_sam.jpg"
            # save_path_gt = save_path_dir + img_name + f"_gt.jpg"
            # plt.imsave(save_path_sam, vis_tensor.view(1024,1024).numpy(), cmap='gray')
            # plt.imsave(save_path_gt, tensor_gt.view(1024,1024).numpy(), cmap='gray')



print(val_metric1.item(),                
                val_metric2.item(),
                val_metric3.item(),
                val_metric4.item(),)
