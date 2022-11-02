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

