import torch
import json
import shutil
import os
import pandas as pd
from PIL import Image
import argparse
from sklearn.cluster import KMeans
from json import dump




def csvs_to_coco_6(dor_path):
    image_dir = os.path.join(dor_path, 'rgb')
    csv_dir = os.path.join(dor_path, 'csv')
    output_directory = os.path.join(dor_path, 'json')
    images = []
    annotations = []
    
    category_mapping = {'ASCUS': 0,
                        'LISL': 1,
                        'ASC-H': 2,
                        'ASCH': 3,
                        'LSIL' : 4,
                        # 'unknown': 3,
    }
    categories = [{'id': category_id, 'name': category_name} for category_name, category_id in category_mapping.items()]


    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = Image.open(os.path.join(image_dir, filename))
            width, height = image.size

            images.append({
                'id': i,
                'width': width,
                'height': height,
                'file_name': filename
            })

        # 检查是否存在与图像同名的 CSV 文件
        csv_filename = filename.replace('.png', '.csv').replace('.jpg', '.csv')
        if csv_filename in os.listdir(csv_dir):
            # 读取 CSV 文件
            df = pd.read_csv(os.path.join(csv_dir, csv_filename))
            # 去掉unknown
            df = df[df['classes_2'] != 'unknown']

            # 遍历 CSV 文件的每一行，添加标注信息
            for _, row in df.iterrows():
                annotations.append({
                    'id': len(annotations) + 1,
                    'image_id': i,
                    'category_id': category_mapping[row['classes_2']],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']],
                    'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']),
                    'iscrowd': 0
                })

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(os.path.join(output_directory, 'coco_6.json'), 'w') as f:
        dump(coco_dict, f)

def csvs_to_coco_5(dor_path):
    image_dir = os.path.join(dor_path, 'rgb')
    csv_dir = os.path.join(dor_path, 'csv')
    output_directory = os.path.join(dor_path, 'json')
    images = []
    annotations = []
    
    category_mapping = {'ASCUS': 0,
                        'LISL': 1,
                        'ASC-H': 2,
                        'ASCH': 3,
                        # 'unknown': 3,
    }
    categories = [{'id': category_id, 'name': category_name} for category_name, category_id in category_mapping.items()]


    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = Image.open(os.path.join(image_dir, filename))
            width, height = image.size

            images.append({
                'id': i,
                'width': width,
                'height': height,
                'file_name': filename
            })

        # 检查是否存在与图像同名的 CSV 文件
        csv_filename = filename.replace('.png', '.csv').replace('.jpg', '.csv')
        if csv_filename in os.listdir(csv_dir):
            # 读取 CSV 文件
            df = pd.read_csv(os.path.join(csv_dir, csv_filename))
            # 去掉unknown
            df = df[df['classes_3'] != 'unknown']

            # 遍历 CSV 文件的每一行，添加标注信息
            for _, row in df.iterrows():
                annotations.append({
                    'id': len(annotations) + 1,
                    'image_id': i,
                    'category_id': category_mapping[row['classes_3']],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']],
                    'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']),
                    'iscrowd': 0
                })

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(os.path.join(output_directory, 'coco_5.json'), 'w') as f:
        dump(coco_dict, f)


def csvs_to_coco_4(dor_path):
    image_dir = os.path.join(dor_path, 'rgb')
    csv_dir = os.path.join(dor_path, 'csv')
    output_directory = os.path.join(dor_path, 'json')
    images = []
    annotations = []
    
    category_mapping = {'ASCUS': 0,
                        'LISL': 1,
                        'ASC-H': 2,
                        # 'unknown': 3,
    }
    categories = [{'id': category_id, 'name': category_name} for category_name, category_id in category_mapping.items()]


    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = Image.open(os.path.join(image_dir, filename))
            width, height = image.size

            images.append({
                'id': i,
                'width': width,
                'height': height,
                'file_name': filename
            })

        # 检查是否存在与图像同名的 CSV 文件
        csv_filename = filename.replace('.png', '.csv').replace('.jpg', '.csv')
        if csv_filename in os.listdir(csv_dir):
            # 读取 CSV 文件
            df = pd.read_csv(os.path.join(csv_dir, csv_filename))
            # 去掉unknown
            df = df[df['classes_4'] != 'unknown']

            # 遍历 CSV 文件的每一行，添加标注信息
            for _, row in df.iterrows():
                annotations.append({
                    'id': len(annotations) + 1,
                    'image_id': i,
                    'category_id': category_mapping[row['classes_4']],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']],
                    'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']),
                    'iscrowd': 0
                })

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(os.path.join(output_directory, 'coco_4.json'), 'w') as f:
        dump(coco_dict, f)


def csvs_to_coco_01(dor_path, label_U_copy):
    image_dir = os.path.join(dor_path, 'rgb')
    csv_dir = os.path.join(dor_path, 'csv')
    output_directory = os.path.join(dor_path, 'json')

    for csv_file in os.listdir(csv_dir):
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)

        if label_U_copy=="U2":
            df['classes_01'] = df['classes_2']
        elif label_U_copy=="U3":
            df['classes_01'] = df['classes_3']
        elif label_U_copy=="U4":
            df['classes_01'] = df['classes_4']
        
        df['classes_01'] = df['classes_01'].replace('unknown', 'knu')
        df['classes_01'] = df['classes_01'].replace(['ASCUS', 'LISL', 'ASC-H', 'ASCH', 'LSIL'], 'knk')

        # 当classes_01列中的值不为knu时且不为knk时，输出错误信息
        if (df['classes_01'] != 'knu').all() and (df['classes_01'] != 'knk').all():
            print('Error: classes_01 column contains values other than knu and knk')
            return
        df.to_csv(csv_path, index=False)
        pass

    images = []
    annotations = []
    
    category_mapping = {'knk': 0,
                        'knu': 1,
    }
    categories = [{'id': category_id, 'name': category_name} for category_name, category_id in category_mapping.items()]


    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = Image.open(os.path.join(image_dir, filename))
            width, height = image.size

            images.append({
                'id': i,
                'width': width,
                'height': height,
                'file_name': filename
            })

        # 检查是否存在与图像同名的 CSV 文件
        csv_filename = filename.replace('.png', '.csv').replace('.jpg', '.csv')
        if csv_filename in os.listdir(csv_dir):
            # 读取 CSV 文件
            df = pd.read_csv(os.path.join(csv_dir, csv_filename))

            # 遍历 CSV 文件的每一行，添加标注信息
            for _, row in df.iterrows():
                annotations.append({
                    'id': len(annotations) + 1,
                    'image_id': i,
                    'category_id': category_mapping[row['classes_01']],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']],
                    'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']),
                    'iscrowd': 0
                })

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(os.path.join(output_directory, 'coco_01.json'), 'w') as f:
        dump(coco_dict, f)

def json():
    U2_json_dir = '../data_base/U2/data_clear/coco_base_all_418'
    U2_json_80 = '../data_base/U2/data_clear/coco_base_test_80'
    U2_json_338 = '../data_base/U2/data_clear/coco_base_train_338'
    U2_json_18 = '../data_base/U2/data_clear/coco_labeled_18'
    U2_json_320 = '../data_base/U2/data_clear/coco_unlabeled_320'

    U3_json_dir = U2_json_dir.replace('U2', 'U3')
    U3_json_80 = U2_json_80.replace('U2', 'U3')
    U3_json_338 = U2_json_338.replace('U2', 'U3')
    U3_json_18 = U2_json_18.replace('U2', 'U3')
    U3_json_320 = U2_json_320.replace('U2', 'U3')

    U4_json_dir = U2_json_dir.replace('U2', 'U4')
    U4_json_80 = U2_json_80.replace('U2', 'U4')
    U4_json_338 = U2_json_338.replace('U2', 'U4')
    U4_json_18 = U2_json_18.replace('U2', 'U4')
    U4_json_320 = U2_json_320.replace('U2', 'U4')


    csvs_to_coco_6(U2_json_dir)
    csvs_to_coco_6(U2_json_80)
    csvs_to_coco_6(U2_json_338)
    csvs_to_coco_6(U2_json_18)
    csvs_to_coco_6(U2_json_320)

    csvs_to_coco_5(U3_json_dir)
    csvs_to_coco_5(U3_json_80)
    csvs_to_coco_5(U3_json_338)
    csvs_to_coco_5(U3_json_18)
    csvs_to_coco_5(U3_json_320)

    csvs_to_coco_4(U4_json_dir)
    csvs_to_coco_4(U4_json_80)
    csvs_to_coco_4(U4_json_338)
    csvs_to_coco_4(U4_json_18)
    csvs_to_coco_4(U4_json_320)

    label_U_copy = "U2"
    csvs_to_coco_01(U2_json_dir, label_U_copy)
    csvs_to_coco_01(U2_json_80, label_U_copy)
    csvs_to_coco_01(U2_json_338, label_U_copy)
    csvs_to_coco_01(U2_json_18, label_U_copy)
    csvs_to_coco_01(U2_json_320, label_U_copy)

    label_U_copy = "U3"
    csvs_to_coco_01(U3_json_dir, label_U_copy)
    csvs_to_coco_01(U3_json_80, label_U_copy)
    csvs_to_coco_01(U3_json_338, label_U_copy)
    csvs_to_coco_01(U3_json_18, label_U_copy)
    csvs_to_coco_01(U3_json_320, label_U_copy)

    label_U_copy = "U4"
    csvs_to_coco_01(U4_json_dir, label_U_copy)
    csvs_to_coco_01(U4_json_80, label_U_copy)
    csvs_to_coco_01(U4_json_338, label_U_copy)
    csvs_to_coco_01(U4_json_18, label_U_copy)
    csvs_to_coco_01(U4_json_320, label_U_copy)

if __name__ == '__main__':
    json()
    pass