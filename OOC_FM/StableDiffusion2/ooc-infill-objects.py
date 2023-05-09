import torch
import torchvision.transforms as T

import PIL
import requests
from io import BytesIO
import os
from datetime import datetime
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt 

import shutil


from pycocotools.coco import COCO

from diffusers import StableDiffusionInpaintPipeline

########################################################################################################
# update these locations
##############################################################################################################

data_directory = "/WorkSpace-2/aroy/data/datasets/coco/coco_2017"
image_directory = os.path.join(data_directory, "images/train2017")
annotation_file = os.path.join(data_directory, "annotations-json/instances_train2017.json")

ooc_directory = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/coco_ooc_dataset/"
ooc_image_directory = os.path.join(ooc_directory, "images")
ooc_ann_directory = os.path.join(ooc_directory, "annotations")

coco_output_directory = "/WorkSpace-2/jha/data/coco_2017_SD2_infilled"
ooc_output_directory = "/WorkSpace-2/jha/data/ooc_coco_2017_SD2_infilled"

########################################################################################################
## generic helper and test methods
##############################################################################################################

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")



def test_inpaint(inpaint_modelpipe): 
    
    test_directory = "test"
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    subdirectory_path = os.path.join(test_directory, current_datetime)
    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
    
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))
    prompt = ""
    image = inpaint_modelpipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    
    
    init_image_path = os.path.join(subdirectory_path, "init_image.png")
    mask_image_path = os.path.join(subdirectory_path, "mask_image.png")
    image_path = os.path.join(subdirectory_path, "filled_image.png")

    init_image.save(init_image_path, format="PNG")
    mask_image.save(mask_image_path, format="PNG")
    image.save(image_path, format="PNG")

##############################################################################################################
## inpaint methods 
##############################################################################################################

def get_inpaint_modelpipe():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda:2")
    return pipe


##############################################################################################################
## run on coco  and ooc
##############################################################################################################

def load_coco_and_sample_images(annotation_file, num_samples):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    selected_image_ids = random.sample(image_ids, num_samples)
    return coco, selected_image_ids


def infill_ooc_samples(pipe, ooc_image_directory, ooc_ann_directory, output_directory, topk, num_samples):
    all_files = os.listdir(ooc_image_directory)
    selected_files = random.sample(all_files, num_samples)
    selected_imageIDs = []
    for filename in selected_files:
        image_path = os.path.join(ooc_image_directory, filename)
        ooc_orig_output_path = os.path.join(output_directory, filename)
        init_image = PIL.Image.open(image_path) 
        filename = filename.split(".")[0]
        infilled_filename = "infilled" + "_" + filename + "_ooc.png"
        infilled_output_path = os.path.join(output_directory, infilled_filename)
        annotation_filename = filename + ".npy"
        annotation_path = os.path.join(ooc_ann_directory, annotation_filename)
        annotation = np.load(annotation_path, allow_pickle=True).item()
        bbox = annotation['ooc_annotation']['bbox']
        y, x, w, h = bbox
        
        mask = PIL.Image.new("1", init_image.size, color=0) # "1" for binary mode, 1 for white
        draw = PIL.ImageDraw.Draw(mask)
        draw.rectangle([x, y, x+w, y+h], fill=1)
        
        original_width, original_height = init_image.size
        print(original_width, original_height)

        masked_image = mask.resize((512, 512))
        image = init_image.resize((512, 512))

        prompt = ""
        infilled_image = pipe(prompt=prompt, image=image, mask_image=masked_image).images[0]            
    
        print(annotation)
        filenameParts = filename.split("_")
        imageID = int(filenameParts[2])
        selected_imageIDs.append(imageID)
        
        oocID = int(filenameParts[-2])
        print(filename, filenameParts, imageID, oocID)
        
        init_image = init_image.resize((original_width, original_height))
        infilled_image = infilled_image.resize((original_width, original_height))
        
        infilled_image.save(infilled_output_path, format="PNG")
        #init_image.save(ooc_orig_output_path, format="PNG")
        
        infill_topk_segments_coco(pipe, coco, imageID, init_image, topk, filename, output_directory)

        
    return selected_imageIDs
        

        
def infill_topk_segments_coco(pipe, coco, image_id, init_image, topk, init_filename, output_path):
    id_to_name_map = {category["id"]: category["name"] for category in coco.dataset["categories"]}    
    object_annotation_ids = coco.getAnnIds(imgIds= [image_id], iscrowd=None)
    anns = coco.loadAnns(object_annotation_ids)
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True) #get the larger annotations first

    original_width, original_height = init_image.size
    print(original_width, original_height)

    for ann in sorted_anns[0:topk]:
        try:
            mask = PIL.Image.new("1", init_image.size, color=0) # "1" for binary mode, 1 for white
            draw = PIL.ImageDraw.Draw(mask)
            print(ann)
            ann_id = ann['id']
            class_id = ann["category_id"]
            class_name = id_to_name_map[class_id]
            class_name = class_name.replace(" ", "-")
            print(f"Image ID: {image_id} ann ID: {ann_id} Class ID: {class_id}, Class Name: {class_name}")
            if "segmentation" in ann:
                for segment in ann["segmentation"]:
                    draw.polygon(segment, fill=1)

            # scale the mask and image and run SD2
            masked_image = mask.resize((512, 512))
            image = init_image.resize((512, 512))
            prompt = ""
            infilled_images = pipe(prompt=prompt, image=image, mask_image=masked_image, num_images_per_prompt=1)

            init_image = init_image.resize((original_width, original_height))
            masked_image = masked_image.resize((original_width, original_height))

            # rescale and write the images back

            filename = "masked" + "_" + init_filename + "_" + str(ann_id) + "_" + str(class_id) + "_" + str(class_name)  + ".png"
            masked_output_path = os.path.join(output_path, filename)
            print(masked_output_path)
            #masked_image.save(masked_output_path, format="PNG")

            for idx, infilled_image in enumerate(infilled_images.images):
                infilled_image = infilled_image.resize((original_width, original_height))
                filename = "infilled" + "_" + init_filename + "_" + str(ann_id) + "_" + str(class_id) + "_" + str(class_name) +  "_" + str(idx) + ".png"
                infilled_output_path = os.path.join(output_path, filename)
                print(infilled_output_path)
                infilled_image.save(infilled_output_path, format="PNG")
        except:
            print(f"Failed: Image ID: {image_id} ann ID: {ann_id} Class ID: {class_id}, Class Name: {class_name}")
        #     output_path = os.path.join(output_directory, str(image_id))
        #     if  os.path.exists(output_path):
        #         shutil.rmtree(output_path)
        
        
        
def infill_coco_samples(pipe, coco, selected_image_ids, topk, image_directory, output_directory):
    for image_id in selected_image_ids:
        selected_image = coco.loadImgs([image_id])[0]
        init_image = PIL.Image.open(os.path.join(image_directory, selected_image["file_name"])).convert("RGB") 
        output_path = os.path.join(output_directory, str(image_id))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created a directory at {output_path}")
        init_filename = selected_image["file_name"] + "_" + str(image_id)
        infill_topk_segments_coco(pipe, coco, image_id, selected_image, topk, init_filename, output_path)
            


##############################################################################################################
# main script 
##############################################################################################################

pipe = get_inpaint_modelpipe()

#test_inpaint(pipe)

coco, selected_image_ids = load_coco_and_sample_images(annotation_file = annotation_file, num_samples  = 1000)

#ignore the randomly selected images and instead use those that were needed from OOC

selected_image_ids = infill_ooc_samples(pipe, ooc_image_directory, ooc_ann_directory, output_directory = ooc_output_directory, topk = 5, num_samples = 100000)

print(f"{len(selected_image_ids)} Selected images were {selected_image_ids}")

#infill_coco_samples(pipe, coco, selected_image_ids, topk=4, image_directory = image_directory, output_directory = coco_output_directory)

