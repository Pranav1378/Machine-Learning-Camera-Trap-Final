import json
import os
import shutil
import argparse
from textwrap import dedent

parser = argparse.ArgumentParser(description='Separate images for json file.')
parser.add_argument('-i', help='Images directory', dest='img_dir', required=True)
parser.add_argument('-j', help='JSON file', dest='json', required=True)
parser.add_argument('-o', help='path to output folder', dest='out',required=True)
parser.add_argument('-m', help='If specified, will execute the move', action='store_true',required=False)

args = parser.parse_args()

images_directory_path = args.img_dir 
json_file = args.json 
output_directory = args.out
really_move = args.m

def check_file_and_dir(file_path, dir_path):
    if not os.path.exists(file_path):
        raise ValueError("file not found")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

check_file_and_dir(json_file, output_directory)
labels = json.load(open(json_file, 'r'))

count = 0
for image in labels['images']:
    file_name = image['file_name']
    file_path = images_directory_path + "/" + file_name
    if os.path.exists(file_path):
        if really_move:
            shutil.move(file_path, output_directory)
        else:
            print("mv %s %s" % (file_path, output_directory))
        count += 1

print("Total images from Json file: ", len(labels['images']))
actually_moved = "" if really_move else "to be"
print("Files %s moved: %s" % (actually_moved, count))
