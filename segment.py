import torchvision
import numpy as np
import torch
import argparse
import cv2

import segmentation_utils
from PIL import Image

# init argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--i', '--input', help='path to input image/video')
args = vars(parser.parse_args())

# get the pretrained model
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set model to eval mode and load onto computation device
model.to(device)

# read image from disk
image = Image.open(args['i'])
# execute forward pass with model
outputs = segmentation_utils.get_segmentation_labels(image, model, device)
outputs = outputs['out']
# get segmented image from model output
segmented_image = segmentation_utils.draw_segmentation_map(outputs)

final_image = segmentation_utils.image_overlay(image, segmented_image)

save_name = f"{args['i'].split('/')[-1].split('.')[0]}"
# save image to disk
cv2.imwrite(f'outputs/{save_name}.jpg', final_image)
