import clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
import datetime
BICUBIC = InterpolationMode.BICUBIC

def get_fused_mask(pil_img, text, sam_predictor, clip_model, args, device, config):
    mask_l = []
    mask_logit_l = []
    mask_logit_origin_l = []

    num_mask = len(config['mask_params']['down_sample'])
    for idx in range(num_mask):
        reset_params(args, config, idx)
        mask, mask_logit, mask_logit_origin, points, labels, num, vis_dict = \
                get_mask(pil_img, text, sam_predictor, clip_model, args, device)
        mask_l.append(mask.astype('float'))
        mask_logit_l.append(mask_logit.astype('float'))
        mask_logit_origin_l.append(mask_logit_origin)

    # mask = sum(mask_l)/num_mask
    mask_logit = sum(mask_logit_l)/num_mask
    mask_logit_origin = sum(mask_logit_origin_l)/num_mask
    mask = mask_logit_origin > sam_predictor.model.mask_threshold
    mask_logit = mask_logit.astype('uint8')
    mask = mask.astype('uint8')

    return mask, mask_logit, mask_logit_origin, points, labels, num, vis_dict

def get_mask(pil_img, text, sam_predictor, clip_model, args, device):

    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        # get heatmap
        sm_mean, sm, vis_map_img, vis_input_img = get_heatmap(pil_img, text, clip_model, args, device)

        # get positive points from individual maps (each sentence in the list), and negative points from the mean map
        points, labels, vis_radius, num = heatmap2points(sm, sm_mean, cv2_img, args)

        if args.clip_use_neg_text:
            sm_mean_n, sm_n, vis_map_img_n, vis_input_img_n = get_heatmap(pil_img, ['background'], clip_model, args, device)
            points_n, labels_n, vis_radius_n, num_n = heatmap2points(sm_n, sm_mean_n, cv2_img, args, attn_thr=args.clip_neg_text_attn_thr )
            points = points + points_n[-num_n:]
            labels = np.concatenate([labels, 1+labels_n[-num_n:]], 0)
            vis_radius = np.concatenate([vis_radius, vis_radius_n[-num_n:]], 0)

        # Inference SAM with points from CLIP Surgery
        sam_predictor.set_image(np.array(pil_img))
        mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True)
        mask_logit_origin = mask_logit_origin[np.argmax(scores)]
        mask = mask_logit_origin > sam_predictor.model.mask_threshold
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy() * 255
        mask = mask.astype('uint8')
        mask_logit = mask_logit.astype('uint8')
    vis_dict = {'vis_map_img': vis_map_img,
                'vis_input_img': vis_input_img, 
                'vis_radius': vis_radius}
        
    return mask, mask_logit, mask_logit_origin, points, labels, num, vis_dict

def get_heatmap(pil_img, text, model, args, device='cuda'):
    
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

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
    # vis_input_img.append(cur_image)
    vis_map_img = []

    sm1 = sm_mean
    side = int(sm1.shape[0] ** 0.5)
    sm1 = sm1.reshape(1, 1, side, side)
    sm1 = torch.nn.functional.interpolate(sm1, (cur_image.shape[0], cur_image.shape[1]), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
    sm1 = (sm1 - sm1.min()) / (sm1.max() - sm1.min())
    sm1 = sm1.cpu().numpy()
    cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
    vis_input_img.append(cur_image.astype('uint8'))
    vis_map_img.append((255*sm1[...,0]).astype('uint8'))

    # if args.recursive>0:
        

    for i in range(args.recursive):
        cur_input_image = Image.fromarray(cur_image.astype(np.uint8))
        cur_input_image = preprocess(cur_input_image).unsqueeze(0).to(device)
        image_features = model.encode_image(cur_input_image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])
        sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
        sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
        sm_mean = sm_norm.mean(-1, keepdim=True)
        sm1 = sm_mean

        side = int(sm1.shape[0] ** 0.5)
        sm1 = sm1.reshape(1, 1, side, side)
        sm1 = torch.nn.functional.interpolate(sm1, (cur_image.shape[0], cur_image.shape[1]), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
        sm1 = (sm1 - sm1.min()) / (sm1.max() - sm1.min())
        sm1 = sm1.cpu().numpy()
        cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
        vis_input_img.append(cur_image.astype('uint8'))
        vis_map_img.append((255*sm1[...,0]).astype('uint8'))


    return sm_mean, sm, vis_map_img, vis_input_img

def get_text_from_img(img_path, pil_img, BLIP_dict=None, model=None, vis_processors=None, device='cuda'):
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

def heatmap2points(sm, sm_mean, cv2_img, args, attn_thr=-1):
    if attn_thr < 0:
        attn_thr = args.attn_thr
    map_l=[]
    p, l, map = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=attn_thr, 
                                                down_sample=args.down_sample,
                                                pt_topk=args.pt_topk) # p: [pos (min->max), neg(max->min)]
    map_l.append(map)
    num = len(p) // 2
    points = p[num:] # negatives in the second half
    labels = [l[num:]]
    vis_radius = [np.linspace(4,1,num)]
    for i in range(sm.shape[-1]):  
        p, l, map = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], cv2_img, t=attn_thr, 
                                                    down_sample=args.down_sample,
                                                    pt_topk=args.pt_topk)
        map_l.append(map)
        num = len(p) // 2
        points = points + p[:num] # positive in first half
        labels.append(l[:num])
        vis_radius.append(np.linspace(2,5,num))
    labels = np.concatenate(labels, 0)
    vis_radius = np.concatenate(vis_radius, 0).astype('uint8')

    return points, labels, vis_radius, num

def printd(str):
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt+'\t '+str)

def reset_params(args, config, idx):
    params = config['mask_params']
    args.down_sample = params['down_sample'][idx]
    args.attn_thr = params['attn_thr'][idx]
    args.pt_topk = params['pt_topk'][idx]
    args.recursive = params['recursive'][idx]
    args.recursive_coef = params['recursive_coef'][idx]
    args.rdd_str = params['rdd_str'][idx]
    args.clip_use_neg_text = params['clip_use_neg_text'][idx]
    args.clip_neg_text_attn_thr = params['clip_neg_text_attn_thr'][idx]

def get_dir_from_args(args, config=None, parent_dir='output_img/'):
    if args.multi_mask_fusion:
        parent_dir += f'cfg_{args.config[:-5]}'
        if args.clip_attn_qkv_strategy!='vv':
            parent_dir += f'_qkv{args.clip_attn_qkv_strategy}'
        num_mask = len(config['mask_params']['down_sample'])
        printd(f'fusing: {parent_dir.split("/")[-1]}')
        for idx in range(num_mask):
            reset_params(args, config, idx)
            cur_param_str = f's{args.down_sample}_thr{args.attn_thr}'
            if args.pt_topk > 0:
                cur_param_str += f'_top{args.pt_topk}'
            if args.recursive > 0:
                cur_param_str += f'_rcur{args.recursive}'
                if args.recursive_coef!=.3:
                    cur_param_str += f'_{args.recursive_coef}'
            if args.clip_use_neg_text:
                cur_param_str += f'_neg{args.clip_neg_text_attn_thr}'
            if args.rdd_str != '':
                cur_param_str += f'_rdd{args.rdd_str}'
            print(f'\t {cur_param_str} ({args})')
        save_path_dir = f'{parent_dir}/'
    else:
        exp_name = f's{args.down_sample}_thr{args.attn_thr}'
        if args.pt_topk > 0:
            exp_name += f'_top{args.pt_topk}'
        if args.recursive > 0:
            exp_name += f'_rcur{args.recursive}'
            if args.recursive_coef!=.3:
                exp_name += f'_{args.recursive_coef}'
        if args.clip_use_neg_text:
            exp_name += f'_neg{args.clip_neg_text_attn_thr}'
        if args.rdd_str != '':
            exp_name += f'_rdd{args.rdd_str}'
        if args.clip_attn_qkv_strategy!='vv':
            exp_name += f'_qkv{args.clip_attn_qkv_strategy}'
        save_path_dir = f'{parent_dir+exp_name}/'
        printd(f'{exp_name} ({args}')

    return save_path_dir

        