'''
check BLIP output & store in file blip_cache/<dataset>
'''

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

BICUBIC = InterpolationMode.BICUBIC



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
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
spec = config['test_dataset']
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = spec['dataset']['args']['root_path_1'].split('/')[2]
output_path = f'blip_cache/{dataset_name}'

## get data
print(f'loading dataset...')
dataset = datasets.make(spec['dataset'])
dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
loader = DataLoader(dataset, batch_size=spec['batch_size'],
                    num_workers=8)
paths_img = dataset.dataset.paths_img
print("dataset size:", len(paths_img))

# for img_path, pair in zip(paths_img, loader):
#     print(img_path,)
#     print( pair['gt'].size())#({'inp':, 'gt':} (1,3,h,w)
# for img_path, pairs in zip(paths_img, loader):
#     tensor_img = pairs['inp']
#     tensor_gt = pairs['gt']
#     pil_img = Image.open(img_path).convert("RGB")
#     cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#     print(cv2_img.shape, tensor_img.shape, tensor_gt.shape)

## load model
# blip (if necessary)
from lavis.models import load_model_and_preprocess
# model_type="pretrain_opt2.7b"
model_type="pretrain_opt6.7b"  #lin
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

import os
open_type = 'a' if os.path.exists(output_path) else 'w'
log_file = open(output_path, open_type)

for img_path, pairs in zip(paths_img, loader):
# for img_path in [
#     'data/COD/TestDataset/COD10K/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-4.jpg',]:
    pil_img = Image.open(img_path).convert("RGB")
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)
            
    text = get_text_from_img(img_path, pil_img, BLIP_dict, BLIP_model, BLIP_vis_processors)
    print(f'img_path:{img_path}\t text:{text}')
    
    log_file.write(f'{text[0]}\n')
    continue
    predictor.set_image(np.array(pil_img))

    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, text, device)

        # Combine features after removing redundant features and min-max norm
        sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]
        sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
        sm_mean = sm_norm.mean(-1, keepdim=True)

        # get positive points from individual maps, and negative points from the mean map
        p, l = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=0.95)
        num = len(p) // 2
        points = p[num:] # negatives in the second half
        labels = [l[num:]]
        for i in range(sm.shape[-1]):
            p, l = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], cv2_img, t=0.95)
            num = len(p) // 2
            points = points + p[:num] # positive in first half
            labels.append(l[:num])
        labels = np.concatenate(labels, 0)

        # Inference SAM with points from CLIP Surgery
        masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True)
        mask = masks[np.argmax(scores)]
        mask = mask.astype('uint8')
        # Visualize the results
        vis = cv2_img[...,0].copy()
        # vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
        vis[mask > 0] = np.array(0, dtype=np.uint8) 
        vis[mask == 0] = np.array(255, dtype=np.uint8)

        # align size of GT mask
        # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        inp_size = 1024
        from torchvision import transforms
        mask_transform = transforms.Compose([
                        transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                        transforms.ToTensor(),
                    ])
        vis_tensor = Image.fromarray(vis)
        vis_tensor = mask_transform(vis_tensor)[0].view(1, 1, inp_size, inp_size)
        from pdb import set_trace
        set_trace()


        result1, result2, result3, result4 = metric_fn(vis_tensor, tensor_gt)
        val_metric1.add(result1.item(), tensor_gt.shape[0])
        val_metric2.add(result2.item(), tensor_gt.shape[0])
        val_metric3.add(result3.item(), tensor_gt.shape[0])
        val_metric4.add(result4.item(), tensor_gt.shape[0])
        print(val_metric1,                
                val_metric2,
                val_metric3,
                val_metric4,)



        ## visualization

        save_path_sam = "output_img/"+img_path.split('/')[-1][:-4]+"_sam.jpg"
        save_path_gt = "output_img/"+img_path.split('/')[-1][:-4]+"_gt.jpg"
        plt.imsave(save_path_sam, vis_tensor.view(1024,1024).numpy(),cmap='gray')
        plt.imsave(save_path_gt, tensor_gt.view(1024,1024).numpy(),cmap='gray')
        # for i, [x, y] in enumerate(points):
        #     cv2.circle(vis, (x, y), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        # print('SAM & CLIP Surgery for texts combination:', text)
        # plt.imshow(vis)
        # plt.show()
        # plt.savefig("./9_sam.jpg")
        # print(save_path)
        # plt.imsave(save_path, vis)


