# Language model for query

import os
import torch
import glob
import numpy as np
import openai
import time
openai.api_key = os.getenv("OPENAI_API_KEY")

captions_filepath = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/blip2_caption/coco_val2014_captions_opt-6.7b-coco.npy"
response_dir      = "/WorkSpace-2/aroy/data/OOC/COCO_OOC/LLM_OP/GPT-4/COCO_val_2014/normal_0_10_gpt4_token_1000_rating"
query_string = " - how normal is this in a scale between 0 to 10? The rating is:"

start_indx = 1024
end_indx   = 10000

os.makedirs(response_dir, exist_ok=True)

# load captions
captions_dict = np.load(captions_filepath, allow_pickle=True)
captions_dict = captions_dict.item()
image_name_list = captions_dict['image_name_list']
caption_list    = captions_dict['caption_list']


# get LLM responses
# gpt_response_list = ['']*len(caption_list)
for indx in range(start_indx, end_indx):


    print(indx)
    image_name = image_name_list[indx]
    caption = caption_list[indx]
    
    if caption: # caption is not empty
        content_str = "\"" + caption + "\"" + query_string
        print (content_str)
        completion = openai.ChatCompletion.create(
          # model="gpt-3.5-turbo",
          model="gpt-4",
          messages=[{"role": "user", "content": content_str}],
          max_tokens=1000
        )

    response = completion.choices[0].message.content

    # write the response
    response_filepath = os.path.join(response_dir, image_name + '.txt')
    with open(response_filepath, "w") as text_file:
        text_file.write(response)

    time.sleep(4)
# response_dict = {'image_name_list': image_name_list, 'response_list':gpt_response_list}
# np.save(response_filepath, response_dict)