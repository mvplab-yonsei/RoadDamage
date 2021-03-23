import json
import cv2
import numpy as np
import os

from collections import defaultdict
index_dic = defaultdict(list)


def extract(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        ID = []
        Image_list = {}  # To retrieve images

        for num in (json_data['images']):
            filename = num['file_name']
            Image_list[num['id']] = filename
            ID.append(num['id'])

        Last_ID = ID[-1]

        GT = np.zeros((720, 1280, 3), np.uint8)  # Create an empty image

        curr_id = 1
        data = 0

        for anno in (json_data['annotations']):

            img_id = anno['image_id']  # Current
            d = {}

            if img_id != curr_id:  # Save & Create new images
                name = str(curr_id) + '.png'

                # Save GT
                anno_path = os.path.join(os.path.expanduser('~'), 'DB', 'Anno_B', name)
                cv2.imwrite(anno_path, GT)

                # Save Image
                image_name = Image_list[curr_id]
                image_path = '/data1/database/data_0310/images'
                image_path = os.path.join(image_path, image_name)

                if not os.path.isfile(image_path):
                    print("NO File:", image_path)
                    raise FileExistsError

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                save_path = os.path.join(os.path.expanduser('~'), 'DB', 'images_B', name)
                cv2.imwrite(save_path, image)
                data += 1
                print('finished: ', curr_id)
                print('data_num: ', data)

                curr_id = img_id
                GT = np.zeros((720, 1280, 3), np.uint8)

                for index, poly in enumerate(anno['polyline']):
                    d[img_id] = poly
                    pts = np.array(list(d.values())).astype(np.int32)
                    pts = pts.reshape(-1, 1, 2)
                    GT = cv2.polylines(GT, [pts], False, (255, 255, 255), 2)  # line thickness: 2

            else:
                curr_id = img_id
                for index, poly in enumerate(anno['polyline']):
                    d[img_id] = poly
                    pts = np.array(list(d.values())).astype(np.int32)
                    pts = pts.reshape(-1, 1, 2)
                    GT = cv2.polylines(GT, [pts], False, (255, 255, 255), 2)  # line thickness: 2

                if curr_id == Last_ID:
                    name = str(curr_id) + '.png'
                    anno_path = os.path.join(os.path.expanduser('~'), 'DB', 'Anno_B', name)
                    cv2.imwrite(anno_path, GT)

                    image_name = Image_list[curr_id]
                    image_path = '/data1/database/data_0310/images'
                    image_path = os.path.join(image_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                    save_path = os.path.join(os.path.expanduser('~'), 'DB', 'images_B', name)
                    cv2.imwrite(save_path, image)

                    print(img_id)


a = '/home/data/0310/anno/CRACK_B.json'  # path to the json file

extract(a)
