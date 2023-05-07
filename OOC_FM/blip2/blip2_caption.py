import os
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import glob
import numpy as np


# paths

img_dir           = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/coco_ooc_dataset/images"
caption_dir       = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/blip2_caption"
captions_filepath = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/blip2_caption/coco_ooc_captions_opt-2.7b.npy"



os.makedirs(caption_dir, exist_ok=True)

# hyperparameters
max_new_tokens=20

# get blip2 captions
# Model

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16) 
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco", torch_dtype=torch.float16) 
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# image captions

image_path_list = glob.glob(os.path.join(img_dir, '*.jpg'))
image_name_list = ['']*len(image_path_list)
caption_list    = ['']*len(image_path_list)

for indx, image_path in enumerate(image_path_list):
    
    print(indx)
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    image_name_list[indx] = image_name
    
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()    
    caption_list[indx] = caption
    
captions_dict = {'image_name_list': image_name_list, 'caption_list':caption_list}
np.save(captions_filepath, captions_dict)