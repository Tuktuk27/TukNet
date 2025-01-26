import sys
import numpy as np
from itertools import product
from typing import List, Tuple
import cv2
import torch
from random import randrange
from utils.SFSORT import SFSORT
from deep_sort_realtime.deepsort_tracker import DeepSort
import os


_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]

__all__ = ["meshgrid"]


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)
    
def decode_nms(input_shape, outputs, keep_top_k, confidence_thresh, iou_thresh, u1x = False):

    grids = []
    strides = []

    ## One of the u1x model has its backbone layer output being switch !!!
    if u1x:
        striding = [8, 16, 32]
        # striding = [8, 16, 32]
        # factor = 13 / 7
        factor = 1
    else:
        striding = [8, 16, 32]
        factor = 1 
    in_w, in_h = input_shape
    hw = [[in_w // s, in_h // s] for s in striding]

    outputs = torch.tensor(outputs)

    outputs = outputs.view(1, torch.sum(torch.prod(torch.tensor(hw), dim=1)), 6)

    for (hsize, wsize), stride in zip(hw, striding):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)

        grids.append(grid)
        shape = grid.shape[:2]

        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1)
    strides = torch.cat(strides, dim=1)

    x_y_c = (outputs[..., 0:2] + grids) * strides * factor
    w_h = torch.exp(outputs[..., 2:4]) * strides * factor
    x_y_min = x_y_c - w_h/2
    x_y_max = x_y_c + w_h/2

    outputs = torch.cat([
        x_y_min,
        x_y_max,
        outputs[..., 4:]
    ], dim=-1)

    # outputs = torch.cat([
    #     (outputs[..., 0:2] + grids) * strides,
    #     torch.exp(outputs[..., 2:4]) * strides,
    #     outputs[..., 4:]
    # ], dim=-1)

    outputs = outputs.numpy().squeeze()

    # NMS from OpenCV

    bboxes = outputs[..., 0:4]
    obj_scores = outputs[...,-2]

    keep_idx = cv2.dnn.NMSBoxes(
    bboxes=bboxes.tolist(),
    scores=obj_scores.tolist(),
    score_threshold=confidence_thresh,
    nms_threshold=iou_thresh,
    eta=1,
    top_k=keep_top_k)  # returns [box_num, class_num]


    if len(keep_idx) > 0:
        NMS_output = outputs[keep_idx]
    else:
        NMS_output = np.empty(shape=(0, outputs.shape[1]))

    out = {}
    for i, box in enumerate(NMS_output):
        out[str(i)] = [np.array(box),'detected', 0]

    return out

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_html_to_txt_and_copy_images(html_dir, image_dir, output_path, classes: list[str], target_size = None):
    os.makedirs(output_path, exist_ok=True)
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.xml')]

    for idx, html_file in enumerate(html_files):
        xml_file = os.path.join(html_dir, html_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # Copy image file if exists
        image_filename = os.path.splitext(html_file)[0] + '.jpg'
        image_src = os.path.join(image_dir, image_filename)
        if os.path.exists(image_src):
            image_dest = os.path.join(output_path, f"image_{idx+1}.jpg")
            img = cv2.imread(image_src)
            if target_size is not None:
                img = cv2.resize(img, target_size)
            cv2.imwrite(image_dest, img)
        else:
            print('doesnt exist')

        # Open TXT file for writing YOLO annotations
        txt_filename = os.path.join(output_path, f"image_{idx+1}.txt")
        with open(txt_filename, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                bb = convert((w, h), (xmin, xmax, ymin, ymax))
                out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

D.2.	Bounded box version between YOLO and COCO format

def bbox_yolo2coco(image_height, image_width, bbox):
    '''
yolo in format of x_center, y_center, width, height (all normalized)
coco in format of xmin, ymin, w, h
'''
    image_height = image_height-1
    image_width = image_width-1
    
    w = bbox[2] * image_height
    
    h = bbox[3] * image_height
    
    x_min = int(max(math.floor((bbox[0]-(bbox[2]/2))*image_width), 0))

    y_min = int(max(math.floor((bbox[1]-(bbox[3]/2))*image_height), 0))

    w = int(min(math.ceil(bbox[2] * image_width), image_width))

    h = int(min(math.ceil(bbox[3] * image_height), image_height))


    return [x_min, y_min, w, h]


def bbox_coco2yolo_v2(image_height, image_width, bbox):
    '''
coco in format of xmin, ymin, h, w
'''
    image_height = image_height-1
    image_width = image_width-1
    center_x = math.floor(((bbox[2]/2) + bbox[0]) / image_width)

    center_y = math.floor(((bbox[3]/2) + bbox[1]) / image_height)

    box_width = math.floor(bbox[2] / image_width)

    box_height = math.floor(bbox[3] / image_height)

    return [center_x, center_y, box_width, box_height]


def mixup(image1, image2, alpha=1):
    """
    Randomly mixes the given list if images with each other
    
    :param images: The images to be mixed up
    :param bboxes: The bounding boxes (labels)
    :param areas: The list of area of all the bboxes
    :param alpha: Required to generate image wieghts (lambda) using beta distribution. In this case we'll use alpha=1, which is same as uniform distribution
    """
    
    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)

    # Weighted Mixup
    mixedup_image = lam*image1 + (1 - lam)*image2

    return mixedup_image

def alternate_mixup(image1, image2):
    """
    Perform mixup by alternating between pixels from two images.

    Args:
    - image1: The first image (numpy array).
    - image2: The second image (numpy array).

    Returns:
    - mixedup_image: The mixed-up image.
    """
    # Ensure images have the same shape
    assert image1.shape == image2.shape, "Image shapes must match"

    mixedup_image = np.empty_like(image1)

    # Alternate between pixels from image1 and image2
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (i + j) % 2 == 0:
                mixedup_image[i, j] = image1[i, j]
            else:
                mixedup_image[i, j] = image2[i, j]

    return mixedup_image

def mixup_data_creation(img_path, lab_path, mixup_function):
    
    reference = []
    to_mixup = []
    
    files = [f for f in os.listdir(img_path) if not f.startswith('.ipynb_checkpoints')]

    # Shuffle the list of image filenames
    random.shuffle(files)
    
    if len(files) % 2 != 0:
        files.pop()
        

    for i in range(0, len(files), 2):
        img_name1 = files[i]
        img_name2 = files[i+1]

        img_path1 = os.path.join(img_path, img_name1)
        img_path2 = os.path.join(img_path, img_name2)

        lab_name1 = f'{os.path.splitext(img_name1)[0]}.txt'
        lab_name2 = f'{os.path.splitext(img_name2)[0]}.txt'

        lab_path1 = os.path.join(lab_path, lab_name1)
        lab_path2 = os.path.join(lab_path, lab_name2)
        
        name_combined = f'{os.path.splitext(img_name1)[0]}_{os.path.splitext(img_name2)[0]}'  # Combined filename

        # Read contents of both text files
        try:
            with open(lab_path1, 'r') as f1:
                content1 = f1.read()
                content1 = content1.strip()
        except FileNotFoundError:
            print(f'Error from {lab_path1 = }')
            content1 = ''

        try:
            with open(lab_path2, 'r') as f2:
                content2 = f2.read()
                content2 = content2.strip()
        except FileNotFoundError:
            print(f'Error from {lab_path2 = }')
            content2 = ''

        # Combine contents and write into a new file
        combined_content = content1 + '\n' + content2

        combined_content = combined_content.strip()

        with open(os.path.join(lab_path, f'{name_combined}.txt'), 'w') as f_combined:
            f_combined.write(combined_content)
        
        # reference_train.append({'image': img_path1, 'label': lab_path1})
        to_mixup.append({'image': img_path2, 'combined name': name_combined})
        

        image1 = cv2.imread(img_path1)
        image2 = cv2.imread(img_path2)
        cv2.imwrite(os.path.join(img_path, f'{name_combined}.jpg'), mixup_function(image1, image2))
        
    print('number of iteration: ', i/2)

def mixup_augmentation(mixup_function, root_im_path: str, root_lab_path: str):
    i = 0

    train_im_path = os.path.join(root_im_path,'train')
    validation_im_path = os.path.join(root_im_path,'val')
    
    train_lab_path = os.path.join(root_lab_path,'train')
    validation_lab_path = os.path.join(root_lab_path,'val')
    
    reference_train = []
    to_mixup_train = []
    mixed_up = []
    
    mixup_data_creation(train_im_path, train_lab_path, mixup_function)
