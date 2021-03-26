import json
import cv2
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', type=str, default="data/annotations/67_sample.json", help='json file path')
parser.add_argument('--image_path', type=str, default='', help='image directory')
parser.add_argument('--train_list', type=str, default='', help='train txt file')
parser.add_argument('--test_liat', type=str, default='', help='test txt file')
opt = parser.parse_args()

image_height = 720
image_width = 1280

img_name_list = []
img_id_list = []
init_id = 0

with open(opt.annotation_path, "r") as json_file:
    json_data = json.load(json_file)

    num_of_image = len(json_data['images'])
    img_info = json_data['images']

    for i in range(num_of_image):
        img_name_list.append(img_info[i]['file_name'])
        img_id_list.append(img_info[i]['id'])

    num_of_bbox = len(json_data['annotations'])
    bbox_info = json_data['annotations']

    for i in range(num_of_bbox):
        if len(bbox_info[i]['bbox']) != 0:
            label_file = open("./database/labels/" + img_name_list[bbox_info[i]['image_id'] - 1].replace(".png.png", ".txt"), 'a')
            img = cv2.imread(os.path.join(opt.image_path,  img_name_list[bbox_info[i]['image_id'] - 1]), 1)

            cv2.imwrite("./database/images/" + img_name_list[bbox_info[i]['image_id'] - 1].replace(".png.png", ".jpg"), img)

            x = bbox_info[i]['bbox'][0]
            y = bbox_info[i]['bbox'][1]
            w = bbox_info[i]['bbox'][2]
            h = bbox_info[i]['bbox'][3]

            cx = x + w / 2
            cy = y + h / 2

            n_cx = cx / image_width
            n_cy = cy / image_height
            n_w = w / image_width
            n_h = h / image_height

            class_id = bbox_info[i]['category_id'] - 1
            
            bbox = str(class_id) + " " + str(n_cx) + " " + str(n_cy) + " " + str(n_w) + " " + str(n_h) + "\n"
            label_file.write(bbox)
            label_file.close()
            

    print(len(bbox_info))

f = open(opt.train_list, 'r')

while True:
    line = f.readline()
    if not line:
        break

    image_name = line.replace("\n", "")
    label_name = image_name.replace(".jpg", ".txt")

    print(image_name)

    shutil.copyfile(os.path.join("./database/images/", image_name), os.path.join("./split/train/", image_name))
    shutil.copyfile(os.path.join("./database/labels/", label_name), os.path.join("./split/train/", label_name))

f.close()

f = open(opt.test_list, 'r')

while True:
    line = f.readline()
    if not line:
        break

    image_name = line.replace("\n", "")
    label_name = image_name.replace(".jpg", ".txt")

    print(image_name)

    shutil.copyfile(os.path.join("./database/images/", image_name), os.path.join("./split/valid/", image_name))
    shutil.copyfile(os.path.join("./database/labels/", label_name), os.path.join("./split/valid/", label_name))

f.close()