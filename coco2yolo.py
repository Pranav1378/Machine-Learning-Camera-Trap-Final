import json
import os
import argparse
from textwrap import dedent

parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-j', help='JSON file', dest='json', required=True)
parser.add_argument('-o', help='path to output folder', dest='out',required=True)
parser.add_argument('-y', help='Dataset yaml file', dest='yaml',required=True)

args = parser.parse_args()

json_file = args.json 
output = args.out
yaml_file = args.yaml


class COCO2YOLO:
    def __init__(self):
        self._check_file_and_dir(json_file, output)
        # self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.labels = json.load(open(json_file, 'r'))
        self.coco_id_name_map = self._categories()
        self.yolo_name_list = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print("Coco id name map %s" % self.coco_id_name_map)
        print("Yolo name list %s" % self.yolo_name_list)
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1.0 / img_w
        dh = 1.0 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            if not 'bbox' in anno:
                print("Skipping image %s" % anno['image_id'])
                continue
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self, yaml_file):
        with open(yaml_file', 'w') as f:
            file_preamble = """
            # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
            train: 
            val: 
            test: 

            # number of classes
            nc: %s

            # class names
            names: %s

            
            """ % (len(self.yolo_name_list), self.yolo_name_list)
            f.write(dedent(file_preamble))
            count = 0
            for cls in self.yolo_name_list:
                f.write('# %s %s\n' % (count, cls))
                count += 1
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = v[0][0].split(".")[0] + ".txt"
            with open(os.path.join(output, file_name), 'w') as f:
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.yolo_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.save_classes(yaml_file)
    c2y.coco2yolo()
