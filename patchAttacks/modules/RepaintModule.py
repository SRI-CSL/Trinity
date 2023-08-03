import torch
from diffusers import StableDiffusionInpaintPipeline

def InitSD2():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    pipe.to("cuda:2")
    return pipe

def getRepaintedImages(imgs, masks, sd2Model):
    repainted_images = []
    for i in range(len(imgs)):
        repainted_image = sd2Model(prompt="", image=imgs[i], mask_image=masks[i]).images[0]
        repainted_images.append(repainted_image)
    
    return repainted_images