import torchvision
import cv2
import torch
import argparse
import time
import segmentation_utils

from PIL import Image

# construct parser
parser = argparse.ArgumentParser()
parser.add_argument('--i', '--input', help='path to input video')
args = vars(parser.parse_args())

# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download or load FCN model
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

# load model onto device
model.eval().to(device)

# load video to cv2 object
cap = cv2.VideoCapture(args['i'])
if not cap.isOpened():
    print('Error while trying to read video. Please check path again..')
# get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# get filename from input argument
save_name = f"{args['i'].split('/')[-1].split('.')[0]}"
# define codec and init VideoWriter object
out = cv2.VideoWriter(f"./outputs/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))
# total frames count
frame_count = 0
# final frames per second
total_fps = 0

# read video until the end
while cap.isOpened():
    # get single frame from video
    ret, frame = cap.read()
