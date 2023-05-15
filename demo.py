import clip
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from segment_anything import sam_model_registry, SamPredictor
from lavis.models import load_model_and_preprocess

### Init CLIP and data
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _ = clip.load("ViT-B/16", device=device)
# model.eval()
img_path = "demo_data/9.jpg"

pil_img = Image.open(img_path).convert("RGB")
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
image = preprocess(pil_img).unsqueeze(0).to(device)
        
BLIP_dict = {"demo_data/9.jpg": 'lizard in the middle',}

def get_text_from_img(img_path, pil_img, ):
    if BLIP_dict.get(img_path) is not None:
        text = [BLIP_dict[img_path]]
    else:
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
        # prepare the image
        # model = model.float()
        image = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
        blip_output2 = model.generate({"image": image, "prompt": "This animal is in the left or in the right or in the middle of the picture? Answer:"})
        print("blip_output2", blip_output2)
        blip_output = model.generate({"image": image})
        print(blip_output)
        # blip_output = blip_output[0].split('-')[0]
        context = [
            ("Image caption.",blip_output),
        ]
        template = "Question: {} Answer: {}."
        question = "Use a word to summary the name of this animal?"
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
        blip_output_forsecond = blip_output_forsecond[0].split('-')[0]
        out_list.append(blip_output_forsecond)
        out_list.append(blip_output2[0])
        out_list = " ".join(out_list)
        text_list = []
        text_list.append(out_list)
        text = text_list
        # text = ["a leaf"]
        print(out_list, blip_output, blip_output2, blip_output_forsecond)
    print('text:', text)
    return text

text = get_text_from_img(img_path, pil_img)

### Explain CLIP via our CLIP Surgery
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()

### Text to points from CLIP Surgery to guide SAM
# Init SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
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
    vis = cv2_img.copy()
    # vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
    vis[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    vis[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    # for i, [x, y] in enumerate(points):
    #     cv2.circle(vis, (x, y), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    print('SAM & CLIP Surgery for texts combination:', text)
    plt.imshow(vis)
    plt.show()
    plt.savefig("./9_sam.jpg")


    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # loader = DataLoader(dataset, batch_size=spec['batch_size'],
    #                     num_workers=8)