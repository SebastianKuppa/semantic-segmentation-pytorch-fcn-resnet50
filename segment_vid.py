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
    if ret:
        # get start time
        start_time = time.time()
        with torch.no_grad():
            # get result for current frame
            outputs = segmentation_utils.get_segmentation_labels(frame, model, device)
        # process frame from video
        segmented_image = segmentation_utils.draw_segmentation_map(outputs['out'])
        final_image = segmentation_utils.image_overlay(frame, segmented_image)

        # get end time
        end_time = time.time()
        # get fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame_count
        frame_count += 1

        # press 'q' on keyboard to exit
        wait_time = max(1, int(fps/4))
        cv2.imshow('image', final_image)
        out.write(final_image)
        if cv2.waitKey(wait_time) and 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture
cap.release()
# close all openCV videos
cv2.destroyAllWindows()

# calc avg fps
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")