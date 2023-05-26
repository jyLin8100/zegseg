

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

BICUBIC = InterpolationMode.BICUBIC

def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)

def get_text_from_img(img_path, pil_img, BLIP_dict=None, model=None, vis_processors=None, ):
    if BLIP_dict.get(img_path) is not None:
        text = [BLIP_dict[img_path]]
    else:
        # prepare the image
        # model = model.float()
        image = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
        blip_output2 = model.generate({"image": image, "prompt": "This animal is in the left or in the right or in the middle of the picture? Answer:"})
        # print("blip_output2", blip_output2)
        blip_output = model.generate({"image": image})
        # print(blip_output)
        # blip_output = blip_output[0].split('-')[0]
        context = [
            ("Image caption.",blip_output[0]),
        ]
        template = "Question: {} Answer: {}."
        # question = "Use a word to summary the name of this animal?"
        question = "Use one single word to summary the name of this animal?"
        prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
        blip_output_forsecond = model.generate({"image": image, "prompt": prompt})
        # blip_output_forsecond = blip_output_forsecond[0].split('_')[0]
        # context1 = [
        #     ("Image caption.", blip_output),
        #     ("Use a word to tell what is this animal?", blip_output_forsecond),
        # ]
        # question2 = "The animal is in the right or in the middle or in the left of this picture?"
        all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']

        out_list = []
        blip_output_forsecond = blip_output_forsecond[0].split('-')[0].replace('\'','')
        out_list.append(blip_output_forsecond)
        out_list.append(blip_output2[0])
        out_list = " ".join(out_list)
        text_list = []
        text_list.append(out_list)
        text = text_list
        # text = ["a leaf"]
        print(f'out_list:{out_list}\n blip_output:{blip_output}\n blip_output2:{blip_output2}\n blip_output_forsecond:{blip_output_forsecond} (prompt: {prompt})')
    print('text:', text)
    return text

## configs
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/mydemo.yaml')
parser.add_argument('--cache_blip_filename', default='COD_GT_woPos') # COD, COD_woPos, COD_GT, COD_GT_woPos, COD_BLIP_GT_woPos
parser.add_argument('--down_sample', type=float, default=2, help='down sample to generate points from CLIP surgery output') 
parser.add_argument('--attn_thr', type=float, default=0.95, help='threshold for CLIP Surgery to get points from attention map') 
parser.add_argument('--pt_topk', type=int, default=-1, help='for CLIP Surgery to get points from attention map, use points of top k highest socre as positive sampling points') 
parser.add_argument('--recursive', type=int, default=0, help='recursive times to use CLIP surgery, to get the point') 
parser.add_argument('--recursive_coef', type=float, default=0.3, help='recursive coefficient to use CLIP surgery, to get the point') 
parser.add_argument('--rdd_str', type=str, default='', help='text for redundant features as input of clip surgery') 

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(f'args:\n{args}\n')
spec = config['test_dataset']
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = spec['dataset']['args']['root_path_1'].split('/')[2]
if args.cache_blip_filename is not None:
    cache_blip_filename = f'blip_cache/{args.cache_blip_filename}'
else:
    cache_blip_filename = f'blip_cache/{dataset_name}'
blip_text_l = []
import os
if os.path.exists(cache_blip_filename):
    cache_blip_file = open(cache_blip_filename, "r")
    for text in cache_blip_file:
        blip_text_l.append(text[:-1].split(','))
        # print(blip_text_l)
    print(f"loading BLIP text output from file: {cache_blip_filename}, length:{len(blip_text_l)}")
    
save_dir_name = f'{cache_blip_filename}/s{args.down_sample}_thr{args.attn_thr}'
if args.pt_topk > 0:
    save_dir_name += f'_top{args.pt_topk}'
if args.recursive > 0:
    save_dir_name += f'_rcur{args.recursive}'
    if args.recursive_coef!=.3:
        save_dir_name += f'_{args.recursive_coef}'
if args.rdd_str != '':
    save_dir_name += f'_rdd{args.rdd_str}'


save_path_dir = f'output_img/{save_dir_name}/'
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
if len(blip_text_l)>0:
    assert data_len==len(blip_text_l)

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
# blip (if necessary)
use_cache_blip = True
if len(blip_text_l)==0:
    use_cache_blip = False
    from lavis.models import load_model_and_preprocess
    # model_type="pretrain_opt2.7b"
    model_type="pretrain_opt6.7b" 
    print(f'loading BLIP ({model_type})...')
    BLIP_model, BLIP_vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type=model_type, is_eval=True, device=device)
    BLIP_dict = {"demo_data/9.jpg": 'lizard in the middle',}

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
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

    if use_cache_blip:
        # text = [blip_text_l[s_i][:-1]]  # lin
        text = blip_text_l[s_i] 
    else:
        text = get_text_from_img(img_path, pil_img, BLIP_dict, BLIP_model, BLIP_vis_processors)

    predictor.set_image(np.array(pil_img))
    print(s_i, text, img_path)

    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(model, [args.rdd_str], device)  # torch.Size([1, 512])

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, text, device)  # torch.Size([x, 512])

        # Combine features after removing redundant features and min-max norm
        sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。

        sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
        sm_mean = sm_norm.mean(-1, keepdim=True)

        vis_input_img = []
        cur_image = np.array(pil_img)
        vis_input_img.append(cur_image)
        if args.recursive>0:
            vis_map_img = []
            sm1 = sm_mean

        for i in range(args.recursive):
            side = int(sm1.shape[0] ** 0.5)
            sm1 = sm1.reshape(1, 1, side, side)
            sm1 = torch.nn.functional.interpolate(sm1, (cur_image.shape[0], cur_image.shape[1]), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
            sm1 = (sm1 - sm1.min()) / (sm1.max() - sm1.min())

            sm1 = sm1.cpu().numpy()
            cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
            vis_input_img.append(cur_image.astype('uint8'))
            vis_map_img.append((255*sm1[...,0]).astype('uint8'))

            cur_input_image = Image.fromarray(cur_image.astype(np.uint8))
            cur_input_image = preprocess(cur_input_image).unsqueeze(0).to(device)
            image_features = model.encode_image(cur_input_image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])
            sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
            sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
            sm_mean = sm_norm.mean(-1, keepdim=True)
            sm1 = sm_mean
            

        # get positive points from individual maps (each sentence in the list), and negative points from the mean map
        map_l=[]
        p, l, map = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=args.attn_thr, 
                                                    down_sample=args.down_sample,
                                                    pt_topk=args.pt_topk) # p: [pos (min->max), neg(max->min)]
        map_l.append(map)
        num = len(p) // 2
        points = p[num:] # negatives in the second half
        labels = [l[num:]]
        vis_radius = [np.linspace(4,1,num)]
        for i in range(sm.shape[-1]):  
            p, l, map = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], cv2_img, t=args.attn_thr, 
                                                        down_sample=args.down_sample,
                                                        pt_topk=args.pt_topk)
            map_l.append(map)
            num = len(p) // 2
            points = points + p[:num] # positive in first half
            labels.append(l[:num])
            vis_radius.append(np.linspace(2,5,num))
        labels = np.concatenate(labels, 0)
        vis_radius = np.concatenate(vis_radius, 0).astype('uint8')

        
        # Inference SAM with points from CLIP Surgery
        masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True)
        mask = masks[np.argmax(scores)]
        mask = mask.astype('uint8')
        # Visualize the results
        vis = cv2_img[...,0].copy()
        vis[mask > 0] = np.array(255, dtype=np.uint8) 
        vis[mask == 0] = np.array(0, dtype=np.uint8)
        vis_pt = np.expand_dims(vis, axis=2).repeat(3, axis=2)
        for i, [x, y] in enumerate(points):
            cv2.circle(vis_pt, (x, y), vis_radius[i], (255, 102, 51) if labels[i] == 1 else (0, 102, 255), vis_radius[i])
            cv2.circle(vis_input_img[0], (x, y), vis_radius[i], (255, 102, 51) if labels[i] == 1 else (0, 102, 255), vis_radius[i])
            if args.recursive>0:
                cv2.circle(vis_map_img[-1], (x, y), vis_radius[i], (255, 102, 51) if labels[i] == 1 else (0, 102, 255), vis_radius[i])
                cv2.circle(vis_input_img[-1], (x, y), vis_radius[i], (255, 102, 51) if labels[i] == 1 else (0, 102, 255), vis_radius[i])
     

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
        print(f'img_path:{img_path}\t text:{text}\t  {result1:.3f} {result2:.3f} {result3:.3f} {result4:.3f} num of points: {num}')
        val_metric1.add(result1.item(), tensor_gt.shape[0])
        val_metric2.add(result2.item(), tensor_gt.shape[0])
        val_metric3.add(result3.item(), tensor_gt.shape[0])
        val_metric4.add(result4.item(), tensor_gt.shape[0])




        ## visualization
        if s_i%1==0 and s_i<10:
            img_name = img_path.split('/')[-1][:-4]
            if args.recursive>0:
                for i in range(len(vis_map_img)):    
                    plt.imsave(save_path_dir + img_name + f'_meanSm{i}.jpg', vis_map_img[i])
                for i in range(len(vis_input_img)):    
                    plt.imsave(save_path_dir + img_name + f'_iptImg{i}.jpg', vis_input_img[i])
                
        
            save_path_sam_pt = save_path_dir + img_name + f"_sam_pt.jpg"
            plt.imsave(save_path_sam_pt, vis_pt)
            # save_path_sam = save_path_dir + img_name + f"_sam.jpg"
            # save_path_gt = save_path_dir + img_name + f"_gt.jpg"
            # plt.imsave(save_path_sam, vis_tensor.view(1024,1024).numpy(), cmap='gray')
            # plt.imsave(save_path_gt, tensor_gt.view(1024,1024).numpy(), cmap='gray')
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)



print(val_metric1.item(),                
                val_metric2.item(),
                val_metric3.item(),
                val_metric4.item(),)
