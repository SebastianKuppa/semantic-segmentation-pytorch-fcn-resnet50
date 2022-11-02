import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

from label_color_map import label_color_map as label_map

# this object will be used to transform all input image to Tensors first, afterwards
# they will be normalized, due to the fact that the dataset, which the CNN model is
# trained on, used the same parameters for normalization
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)


def get_segmentation_labels(image, model, device):
    """
    transforms image to a tensor and applies the FCN model to it
    :param image: input image
    :param model: FCN resnet50 model
    :param device: cuda or cpu
    :return: output tensor
    """
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)

    # uncomment following lines for more information
    # print(type(outputs))
    # print(outputs['out'].shape)
    # print(outputs)

    return outputs


def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
