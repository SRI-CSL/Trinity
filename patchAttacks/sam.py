from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


sam = sam_model_registry["vit_h"](checkpoint="sam_models/sam_vit_h_4b8939.pth")
sam.to(device='cuda')
#

points = [[2100,2600],[1600,2100],[1100,1800]]

for i in range(3):
	image  = cv2.imread('apricot_ds/imgs/img_{}.png'.format(i))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print("GT shape : ", image.shape)	
	#sam = sam_model_registry["vit_h"](checkpoint="sam_models/sam_vit_h_4b8939.pth")
	#sam.to(device='cuda')
	#
	predictor = SamPredictor(sam)
	predictor.set_image(image)
	
	input_point = np.array([points[i]])
	input_label = np.array([1])
	
	masks, scores, logits = predictor.predict(
	    point_coords=input_point,
	    point_labels=input_label,
	    multimask_output=True,
	)
	
	#print(image.shape)
	#print(masks.shape)
	
	#for i, (mask, score) in enumerate(zip(masks, scores)):
	#    plt.figure(figsize=(10,10))
	#    plt.imshow(image)
	#    show_mask(mask, plt.gca())
	#    show_points(input_point, input_label, plt.gca())
	#    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
	#    plt.axis('off')
	#    #plt.savefig('mask_{}.png'.format(i))
	#    plt.show()
	#    plt.savefig('apricot_ds/results/mask_{}_{}.png'.format(2,i))
	
	for j, (mask, score) in enumerate(zip(masks, scores)):
	    plt.imshow(mask, cmap='gray')
	    print("Mask Shape : ", mask.shape)
	    plt.show()
	    plt.axis('off')
	    plt.savefig('apricot_ds/results/only_mask_{}_{}.png'.format(i,j))
	
	#masks, _, _ = predictor.predict()
	#print(masks.shape)
	#mask_generator = SamAutomaticMaskGenerator(sam)
	#masks = mask_generator.generate('apricot_imgs/img_0.png')
	#print(masks)
