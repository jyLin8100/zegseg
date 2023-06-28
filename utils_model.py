from asyncio import wait
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
import datetime
BICUBIC = InterpolationMode.BICUBIC

from scipy.ndimage import gaussian_filter
import scipy
eps = 1e-7

def get_fused_mask(pil_img, text, sam_predictor, clip_model, args, device, config):

    # get list of masks
    mask_l = []
    mask_logit_origin_l = []

    num_mask = len(config['mask_params']['down_sample'])
    for idx in range(num_mask):
        reset_params(args, config, idx)
        mask, _, mask_logit_origin, points, labels, num, vis_dict = \
                get_mask(pil_img, text, sam_predictor, clip_model, args, device)
        mask_l.append(mask.astype('float'))
        mask_logit_origin_l.append(mask_logit_origin)

    # fusion
    if args.multi_mask_fusion_strategy=='entropy':
        mask_logit_l = [F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy() for mask_logit_origin in mask_logit_origin_l]
        mask_entropy_l = [-mask_logit*np.log2(mask_logit) for mask_logit in mask_logit_l]
        mask_entropy_l = [ 1-mask_entropy for mask_entropy in mask_entropy_l ]
        mask_sum = sum(mask_entropy_l)+eps
        mask_w = [ mask_entropy/mask_sum for mask_entropy in mask_entropy_l]
        for i in range(num_mask):
            mask_logit_origin_l[i] *= mask_w[i]
        mask_logit_origin = sum(mask_logit_origin_l)
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
        mask = mask_logit_origin > sam_predictor.model.mask_threshold
    elif args.multi_mask_fusion_strategy=='entropy2':
        mask_logit_l = [F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy() for mask_logit_origin in mask_logit_origin_l]
        mask_entropy_l = [ -mask_logit*np.log2(mask_logit) for mask_logit in mask_logit_l ]
        mask_entropy_l = [ 1-mask_entropy for mask_entropy in mask_entropy_l ]
        mask_sum = sum(mask_entropy_l)+eps
        mask_w = [ mask_entropy/mask_sum for mask_entropy in mask_entropy_l]
        mask_l_w = []
        for i in range(num_mask):
            mask_l_w.append(mask_w[i]*mask_l[i])
        mask_logit = sum(mask_l_w)
        mask = mask_logit > 0.5
    else:   # avg
        # mask_logit = sum(mask_logit_l)/num_mask
        mask_logit_origin = sum(mask_logit_origin_l)/num_mask  # 
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
        mask = mask_logit_origin > sam_predictor.model.mask_threshold

    mask = mask.astype('uint8')
    # print(mask[0,0])
    mask_logit *= 255
    mask_logit = mask_logit.astype('uint8')

    return mask, mask_logit, points, labels, num, vis_dict


def fuse_mask(mask_logit_origin_l, sam_thr, fuse='avg'):

    num_mask = len(mask_logit_origin_l)
    if fuse=='avg':  
        mask_logit_origin = sum(mask_logit_origin_l)/num_mask  # 
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
        mask = mask_logit_origin > sam_thr

    mask = mask.astype('uint8')
    mask_logit *= 255
    mask_logit = mask_logit.astype('uint8')

    return mask, mask_logit

def get_mask(pil_img, text, sam_predictor, clip_model, args, device='cuda'):
    
    vis_input_img = []
    vis_map_img = []  # map applied to img in next iteration
    vis_clip_sm_img = []  # heatmap from clip surgery
    sm_l = []
    sm_mean_l = []

    vis_mask_l = []
    vis_mask_logit_l = []
    mask_logit_origin_l = []
    vis_radius_l = []
    points_l = []
    labels_l = []
    num_l = []
    vis_mask0_l = []  # mask before post refine. 
    bbox_list = []  # for port_mode =='MaxIOUBoxSAMInput':

    ori_image = np.array(pil_img)

    sam_predictor.set_image(ori_image)

    cur_image = ori_image
    vis_input_img.append(cur_image.astype('uint8'))
    with torch.no_grad():
        for i in range(args.recursive+1):
            sm, sm_mean, sm_logit = clip_surgery(cur_image, text, clip_model, args, device='cuda')
            if i==0:    original_sm_norm = sm_logit[..., 0]

            # get positive points from individual maps (each sentence in the list), and negative points from the mean map
            points, labels, vis_radius, num = heatmap2points(sm, sm_mean, cur_image, args)
            if (args.use_origin_neg_points or args.add_origin_neg_points)  and i==0:
                ori_neg_points = points[:num]
                ori_neg_labels = labels[:num]
                ori_neg_vis_radius = vis_radius[:num]
            
            if args.use_origin_neg_points:
                points = ori_neg_points + points[num:]
                labels = np.concatenate([ori_neg_labels, labels[num:]], 0)
                vis_radius = np.concatenate([ori_neg_vis_radius, vis_radius[num:]], 0)

            
            if args.add_origin_neg_points:
                points = ori_neg_points + points
                labels = np.concatenate([ori_neg_labels, labels], 0)
                vis_radius = np.concatenate([ori_neg_vis_radius, vis_radius], 0)

            # Inference SAM with points from CLIP Surgery
            # 1 use fused attn map(sm1) as mask input. -> hm 某些低亮度低也会被纳入mask。
            if args.post_mode=='smSAMInput':
                if i==0:
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                else:
                    h, w = sm1.shape[0:2]
                    _sm = torch.from_numpy(sm1.reshape(1, 1, h, w))
                    _sm = torch.nn.functional.interpolate(_sm, (256, 256), mode='bilinear')[0, 0, :, :].unsqueeze(0)
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,
                            mask_input=_sm)
            # 2 use logit of first output as mask input. 还是不能解决不稳定性，受到第一次mask的影响
            elif args.post_mode=='PostLogit':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                # Cascaded Post-refinement-1: use low-res mask
                best_idx = np.argmax(scores)
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                            point_coords=np.array(points),
                            point_labels=labels,
                            mask_input=logits[best_idx: best_idx + 1, :, :], 
                            multimask_output=True,
                            return_logits=True)
            # 3  use mask in last iter.
            elif args.post_mode=='LogitSAMInput':
                if i==0:
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                else:
                    best_idx = np.argmax(scores)
                    mask_logit_origin, scores, logits,  = sam_predictor.predict(
                            point_coords=np.array(points),
                            point_labels=labels,
                            mask_input=logits[best_idx: best_idx + 1, :, :], 
                            multimask_output=True,
                            return_logits=True)
            # 4. get box from first output as box input.
            elif args.post_mode == 'PostBox':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                best_idx = np.argmax(scores)
                mask = mask_logit_origin[best_idx] > sam_predictor.model.mask_threshold
                y, x = np.nonzero(mask)
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                    point_coords=np.array(points),
                    point_labels=labels,
                    box=input_box[None, :],
                    # mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)
            # 4. get box from first output as box input, but no points input which cuz discontinuous part
            elif args.post_mode == 'PostBoxWoPt':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                best_idx = np.argmax(scores)
                mask = mask_logit_origin[best_idx] > sam_predictor.model.mask_threshold
                y, x = np.nonzero(mask)

                if len(x)!=0:
                    x_min = x.min() 
                    x_max = x.max() 
                    y_min = y.min()
                    y_max = y.max()
                else:
                    x_min = 0
                    x_max = mask_logit_origin[0].shape[1]
                    y_min = 0
                    y_max = mask_logit_origin[0].shape[0]
                input_box = np.array([x_min, y_min, x_max, y_max])
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                    # point_coords=np.array(points),
                    # point_labels=labels,
                    box=input_box[None, :],
                    # mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)
            # 5. enlarge the box to 1.1x 
            elif args.post_mode == 'PostBoxWoPt1.1':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                best_idx = np.argmax(scores)
                mask = mask_logit_origin[best_idx] > sam_predictor.model.mask_threshold

                vis_mask0_l.append(mask.astype('uint8'))
                h, w = mask_logit_origin[0].shape
                y, x = np.nonzero(mask)
                if len(x)!=0:
                    x_min = x.min() 
                    x_max = x.max() 
                    y_min = y.min()
                    y_max = y.max()
                else:
                    x_min = 0
                    x_max = mask_logit_origin[0].shape[1]
                    y_min = 0
                    y_max = mask_logit_origin[0].shape[0]
                x_delta = (x_max - x_min) * 0.05
                y_delta = (y_max - y_min) * 0.05
                x_min = max(0, x_min-x_delta)
                y_min = max(0, y_min-y_delta)
                x_max = min(w, x_max+x_delta)
                y_max = min(h, y_max+y_delta)

                input_box = np.array([x_min, y_min, x_max, y_max])
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                    # point_coords=np.array(points),
                    # point_labels=labels,
                    box=input_box[None, :],
                    # mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)
            # 6. enlarge the box to 1.1x, also use mask to avoid seg the bg in the box
            elif args.post_mode == 'PostBoxMaskWoPt1.1':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                best_idx = np.argmax(scores)
                mask = mask_logit_origin[best_idx] > sam_predictor.model.mask_threshold

                vis_mask0_l.append(mask.astype('uint8'))
                h, w = mask_logit_origin[0].shape
                y, x = np.nonzero(mask)
                if len(x)!=0:
                    x_min = x.min() 
                    x_max = x.max() 
                    y_min = y.min()
                    y_max = y.max()
                else:
                    x_min = 0
                    x_max = mask_logit_origin[0].shape[1]
                    y_min = 0
                    y_max = mask_logit_origin[0].shape[0]
                x_delta = (x_max - x_min) * 0.05
                y_delta = (y_max - y_min) * 0.05
                x_min = max(0, x_min-x_delta)
                y_min = max(0, y_min-y_delta)
                x_max = min(w, x_max+x_delta)
                y_max = min(h, y_max+y_delta)

                input_box = np.array([x_min, y_min, x_max, y_max])
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                    # point_coords=np.array(points),
                    # point_labels=labels,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)
            
            # 6. enlarge the box to 1.1x, also use inflated mask to avoid seg the bg in the box (cuz some discontinuous part. but -> include too much bg area)
            elif args.post_mode == 'PostBoxMaskWoPtInflate1.1':
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                best_idx = np.argmax(scores)
                mask = mask_logit_origin[best_idx] > sam_predictor.model.mask_threshold

                vis_mask0_l.append(mask.astype('uint8'))
                h, w = mask_logit_origin[0].shape
                y, x = np.nonzero(mask)
                if len(x)!=0:
                    x_min = x.min() 
                    x_max = x.max() 
                    y_min = y.min()
                    y_max = y.max()
                else:
                    x_min = 0
                    x_max = mask_logit_origin[0].shape[1]
                    y_min = 0
                    y_max = mask_logit_origin[0].shape[0]
                x_delta = (x_max - x_min) * 0.05
                y_delta = (y_max - y_min) * 0.05
                x_min = max(0, x_min-x_delta)
                y_min = max(0, y_min-y_delta)
                x_max = min(w, x_max+x_delta)
                y_max = min(h, y_max+y_delta)

                input_box = np.array([x_min, y_min, x_max, y_max])

                logits[best_idx] = scipy.ndimage.grey_dilation(logits[best_idx],  size=(3,3))
                mask_logit_origin, scores, logits,  = sam_predictor.predict(
                    # point_coords=np.array(points),
                    # point_labels=labels,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)

            # 7. use max iou box from last mask
            elif args.post_mode =='MaxIOUBoxSAMInput':
                if i==0:
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                else:
                    mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), box=bbox_list[i-1][None, :],multimask_output=True, return_logits=True)
                mask = mask_logit_origin[np.argmax(scores)] > sam_predictor.model.mask_threshold

                #计算最可能的bbox
                contours, _ = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bboxes = []
                overlaps = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = np.array([x, y, x + w, y + h])
                    bboxes.append(bbox)
                    overlap = (w * h) / np.sum(mask)
                    overlaps.append(overlap)
                bboxes = np.array(bboxes)
                overlaps = np.array(overlaps)
                max_overlap_idx = np.argmax(overlaps)
                max_bbox = bboxes[max_overlap_idx]
                scaled_bbox = max_bbox.copy()
                scaled_bbox[:2] -= np.floor((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int)
                scaled_bbox[2:] += np.ceil((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int)
                bboxes[max_overlap_idx] = scaled_bbox
                bboxes = bboxes[max_overlap_idx]
                bbox_list.append(bboxes)
                # # 绘制边界框矩形
                # image_bgr = cv2.cvtColor(cur_image.astype('uint8'), cv2.COLOR_RGB2BGR)
                # x_min, y_min, x_max, y_max = bboxes.astype(int)
                # cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # vis_input_img.append(image_rgb.astype('uint8'))
            else:
                mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)


            mask_logit_origin = mask_logit_origin[np.argmax(scores)]
            mask_logit_origin_l.append(mask_logit_origin)
            mask = mask_logit_origin > sam_predictor.model.mask_threshold
            mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy() 

            # update input image for next iter
            sm1 = sm_logit
            if args.use_fuse_mask_hm:
                mask_logit1 = mask_logit
                mask_logit1[mask_logit1<0.5] = 0
                mask_logit1 = np.expand_dims(mask_logit1, axis=2)
                sm1 = np.clip(  mask_logit1 + sm1, 0, 1)

            if args.use_dilation:
                sm1d = sm1.reshape(cur_image.shape[0], cur_image.shape[1])
                sm1d = scipy.ndimage.grey_dilation(sm1d, size=(args.dilation_k, args.dilation_k))
                sm1 = np.clip(sm1d.reshape(cur_image.shape[0], cur_image.shape[1], 1), 0, 1)
            
            if args.use_origin_img:
                cur_image = ori_image
            
            if args.use_blur and args.use_blur1:
                exit()
            if args.use_blur:
                cur_image_bg = np.clip(gaussian_filter(cur_image, sigma=args.recursive_blur_gauSigma),0,255) * (1-sm1)
                cur_image_fg = cur_image * sm1
                cur_image = (cur_image_fg + cur_image_bg) * args.recursive_coef + cur_image * (1-args.recursive_coef)
            elif args.use_blur1:  # EMA blur
                cur_image_bg = np.clip(gaussian_filter(ori_image, sigma=args.recursive_blur_gauSigma),0,255) * (1-sm1)
                cur_image_fg = ori_image * sm1
                cur_image = (cur_image_fg + cur_image_bg) * args.recursive_coef + cur_image * (1-args.recursive_coef)
            else:
                if args.clipInputEMA:
                    cur_image = ori_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
                else:
                    cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
            
            # collect for visualization
            vis_input_img.append(cur_image.astype('uint8'))
            vis_clip_sm_img.append((255*sm_logit[...,0]).astype('uint8'))
            vis_map_img.append((255*sm1[...,0]).astype('uint8'))
            sm_l.append(sm)
            sm_mean_l.append(sm_mean)

            vis_mask_l.append(mask.astype('uint8'))
            vis_mask_logit_l.append((mask_logit * 255).astype('uint8'))
            vis_radius_l.append(vis_radius)
            points_l.append(points)
            labels_l.append(labels)
            num_l.append(num)



        vis_dict = {'vis_map_img': vis_map_img,
                'vis_clip_sm_img': vis_clip_sm_img,
                'vis_input_img': vis_input_img, 
                'vis_radius': vis_radius_l[-1],
                'original_sm_norm': original_sm_norm,
                'vis_mask_l': vis_mask_l,
                'vis_mask_logit_l': vis_mask_logit_l,
                'vis_radius_l': vis_radius_l,
                'points_l': points_l,
                'labels_l': labels_l,
                'num_l': num_l,
                'vis_mask0_l': vis_mask0_l,
                'mask_logit_origin_l': mask_logit_origin_l,}
        
    return vis_mask_l[-1], vis_mask_logit_l[-1], mask_logit_origin, points_l[-1], labels_l[-1], num_l[-1], vis_dict
    return mask, mask_logit, mask_logit_origin, points, labels, num, vis_dict

    return sm_mean_l[-1], sm_l[-1], vis_map_img, vis_input_img, original_sm_norm


def clip_surgery(np_img, text, model, args, device='cuda'):
    pil_img = Image.fromarray(np_img.astype(np.uint8))
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

    # expand similarity map to original image size, normalize. to apply to image for next iter
    h, w = np_img.shape[:2]
    def _norm_sm(_sm, h, w):
        side = int(_sm.shape[0] ** 0.5)
        _sm = _sm.reshape(1, 1, side, side)
        _sm = torch.nn.functional.interpolate(_sm, (h, w), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
        _sm = (_sm - _sm.min()) / (_sm.max() - _sm.min())
        _sm = _sm.cpu().numpy()
        return _sm
    sm1 = sm_mean
    sm1 = _norm_sm(sm1, h, w) 
    return sm, sm_mean, sm1


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

def heatmap2points(sm, sm_mean, np_img, args, attn_thr=-1):
    cv2_img = cv2.cvtColor(np_img.astype('uint8'), cv2.COLOR_RGB2BGR)
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
    # points = [] # to see the results of only using positive point prompts
    # labels = [[]]
    # vis_radius = [np.linspace(4,1,0)]

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
        if args.multi_mask_fusion_strategy!='avg':
            parent_dir += f'_fuse{args.multi_mask_fusion_strategy}'
        exp_name += f'_sigma{args.recursive_blur_gauSigma}'

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

        if args.use_origin_img:
            exp_name += f'_oriImg'
        if args.use_dilation:
            exp_name += f'_dilation{args.dilation_k}'
        if args.use_blur:
            exp_name += f'_blur'
            exp_name += f'_sigma{args.recursive_blur_gauSigma}'
        if args.use_blur1:
            exp_name += f'_blur1'
            exp_name += f'_sigma{args.recursive_blur_gauSigma}'
        if args.clipInputEMA:  # darken
            exp_name += f'_clipInputEMA'
        if args.use_fuse_mask_hm:
            exp_name += f'_fuseMask'
        if args.use_origin_neg_points:
            exp_name += f'_useOriNegPt'
        if args.add_origin_neg_points:
            exp_name += f'_addOriNegPt'
        if args.post_mode !='':
            exp_name += f'_post{args.post_mode}'
        if args.multi_head:
            exp_name += f'_mulHead'

        save_path_dir = f'{parent_dir+exp_name}/'
        printd(f'{exp_name} ({args}')

    return save_path_dir

        