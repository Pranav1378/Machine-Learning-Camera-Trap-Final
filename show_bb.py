# from PIL import Image
# img = Image.open("cct_test/585a627b-23d2-11e8-a6a3-ec086b02610b.jpg")
# img.show(img)

import cv2
import os
import re
import ast
import argparse

parser = argparse.ArgumentParser(description='Show yolo bounding boxes.') 
parser.add_argument('-i', help='Images directory', dest='img_dir', required=True)
parser.add_argument('-a', help='Annotations directory', dest='ann_dir',required=True)
parser.add_argument('-y', help='Dataset yaml file', dest='yaml',required=True)

args = parser.parse_args()

images_directory_path = args.img_dir 
annotations_directory_path = args.ann_dir 
yaml_file = args.yaml

# annotations_directory_path = "cct_images_annotations"
# images_directory_path = "cct_images"

# annotations_directory_path = "cct_test"
# images_directory_path = "cct_test"

def parse_yolo_bbox(bbox_line, image_width, image_height):
    class_id, x_center, y_center, bbox_width, bbox_height = map(float, bbox_line.split())
    
    x_min = int((x_center - bbox_width / 2) * image_width)
    y_min = int((y_center - bbox_height / 2) * image_height)
    x_max = int((x_center + bbox_width / 2) * image_width)
    y_max = int((y_center + bbox_height / 2) * image_height)
    
    return class_id, x_min, y_min, x_max, y_max

def draw_bboxes(image, bbox_file_path):
    with open(bbox_file_path, 'r') as f:
        bbox_lines = f.read().splitlines()
    
    image_height, image_width = image.shape[:2]
    
    for line in bbox_lines:
        class_id, x_min, y_min, x_max, y_max = parse_yolo_bbox(line, image_width, image_height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return (image, class_id)

def display_file(image_path, bbox_file_path, categories_list):
    image = cv2.imread(image_path)
    (image_with_bboxes, class_id) = draw_bboxes(image.copy(), bbox_file_path)
    class_id = int(class_id)
    category = categories_list[class_id]

    origin = (100, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255) # BGR
    thickness = 2
    
    image_with_bboxes = cv2.putText(image_with_bboxes, '%s (%s)' % (category, class_id),
                                    origin, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Image with Bounding Boxes', image_with_bboxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def list_files_in_directory(directory_path):
    file_list = []
    for filename in os.listdir(directory_path):
        _, extension = os.path.splitext(filename)
        if os.path.isfile(os.path.join(directory_path, filename)) and extension == '.jpg':
            file_list.append(filename)
    return file_list

def read_dataset_yaml(yaml_file):
    file = open(yaml_file, 'r')
    lines = file.readlines()
    for line in lines:
        match = re.match(r'^\s*names:\s*(.*)', line)
        if match:
            categories = match.group(1)
    file.close()
    categories = ast.literal_eval(categories)
    return categories
    
files = list_files_in_directory(images_directory_path)

# Read dataset.yaml to get the categories
categories_list = read_dataset_yaml(yaml_file)

for image_file in files:
    base_name, _ = os.path.splitext(image_file)
    bbox_file = base_name + ".txt"
    image_path = images_directory_path + "/" + image_file
    bbox_file_path = annotations_directory_path + "/" + bbox_file
    print("File: %s, BBOX: %s" % (image_path, bbox_file_path))
    if os.path.exists(image_path) and os.path.exists(bbox_file_path):
        print(image_path, bbox_file_path)
        display_file(image_path, bbox_file_path, categories_list)
    

# image_path = 'cct_test/585a627b-23d2-11e8-a6a3-ec086b02610b.jpg'
# bbox_file_path = 'cct_test/585a627b-23d2-11e8-a6a3-ec086b02610b.txt'
# display_file(image_path, bbox_file_path)

