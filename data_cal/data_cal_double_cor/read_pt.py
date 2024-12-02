# {'image_id': 0, 'category_id': 0, 'bbox': [97.10806274414062, 22.93488883972168, 34.50230407714844, 55.322998046875], 
# 'score': 0.9999943971633911, 'proposal_bbox': [104.99889373779297, 32.15027618408203, 30.76380157470703, 50.87652587890625], 
# 'cls_sc_save': [9.031961441040039, 0.0, 0.0], 'input_features_save': [0.0, 0.07759267091751099, ......， 0.0, 0.0], 
# 'file_path': '/home/yagao/workspace/openset/mydete/data/coco_unlabeled_200/rgb/ANCHFOV-32_TCGA-AR-A1AR-01Z-00-DX1_left-16820_top-29839_bottom-30142_right-17040.png'}

import torch
import json
import shutil
import os
import pandas as pd
from PIL import Image
import argparse
import numpy as np
from json import dump
from json import loads as jsloads


def load_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data_new = jsloads(line)
            data.append(data_new[0])
    return data

def random_crop(array, target_shape=(1, 1024, 10, 10)):
    # 获取原始数组的形状
    original_shape = array.shape

    # 计算裁剪的起始点
    start_dim3 = np.random.randint(0, original_shape[2] - target_shape[2] + 1)
    start_dim4 = np.random.randint(0, original_shape[3] - target_shape[3] + 1)

    # 进行裁剪
    cropped_array = array[:, :, start_dim3:start_dim3+target_shape[2], start_dim4:start_dim4+target_shape[3]]

    return cropped_array

def get_pr_score(file_path, data_json, coreset_features_map):
    pr_score = {}
    for file in file_path:
        pr_score[file] = {}
        
        pr_score_l2_norm = []

        # 获取data_json中file_path为file的features_map
        for item in data_json:
            if item["file_path"] == file:
                pr_score[file]["features_map"] = random_crop(np.array(item["features_map"]))
                break

        # 计算data_json中file_path为file的features_map与coreset_features_map中各个features_map的L2范数
        for coreset_features in coreset_features_map:
            pr_score_l2_norm.append(np.linalg.norm(pr_score[file]["features_map"] - coreset_features))

        pr_score[file]["pr_score_final"] = np.min(pr_score_l2_norm)
    return pr_score

def file_copy(file_path_top, dst_path, old_base_dir_path, dst_path_unlabeled, old_base_dir_path_unlabeled):
    # 当文件夹不存在时，创建文件夹
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if not os.path.exists(dst_path + "/rgb"):
        os.makedirs(dst_path + "/rgb")
    if not os.path.exists(dst_path + "/csv"):
        os.makedirs(dst_path + "/csv")
    if not os.path.exists(dst_path + "/json"):
        os.makedirs(dst_path + "/json")
    if not os.path.exists(dst_path_unlabeled):
        os.makedirs(dst_path_unlabeled)
    if not os.path.exists(dst_path_unlabeled + "/rgb"):
        os.makedirs(dst_path_unlabeled + "/rgb")
    if not os.path.exists(dst_path_unlabeled + "/csv"):
        os.makedirs(dst_path_unlabeled + "/csv")
    if not os.path.exists(dst_path_unlabeled + "/json"):
        os.makedirs(dst_path_unlabeled + "/json")
    # 显示文件夹路径
    print("dst_path: ", dst_path)
    print("dst_path_unlabeled: ", dst_path_unlabeled)
    print("old_base_dir_path: ", old_base_dir_path)
    print("old_base_dir_path_unlabeled: ", old_base_dir_path_unlabeled)
    # 输出各个文件夹中的文件数量
    print("dst_path: ", len(os.listdir(dst_path + "/rgb")))
    print("dst_path_unlabeled: ", len(os.listdir(dst_path_unlabeled + "/rgb")))
    print("old_base_dir_path: ", len(os.listdir(old_base_dir_path + "/rgb")))
    print("old_base_dir_path_unlabeled: ", len(os.listdir(old_base_dir_path_unlabeled + "/rgb")))

    # 将对应位置的图像复制到指定文件
    for file_path in file_path_top:
        shutil.copy(file_path, dst_path + "/rgb")
    
    # 遍历old_base_dir_path_unlabeled中的所有图像，当其完整路径不在file_path_top中时，将其复制到指定文件夹dst_path_unlabeled
    for file_path in os.listdir(old_base_dir_path_unlabeled + "/rgb"):
        if old_base_dir_path_unlabeled + "/rgb/" + file_path not in file_path_top:
            shutil.copy(old_base_dir_path_unlabeled + "/rgb/" + file_path, dst_path_unlabeled + "/rgb")
    
    file_path_top_csv = file_path_top.copy()
    # 将file_path_top_20中各项的"/rgb/"替换为"/csv"
    for i in range(len(file_path_top)):
        file_path_top_csv[i] = file_path_top_csv[i].replace("/rgb/", "/csv/")
        file_path_top_csv[i] = file_path_top_csv[i].replace(".png", ".csv").replace('.jpg', '.csv')

    # 将对应位置的csv文件复制到指定文件夹
    for file_path in file_path_top_csv:
        shutil.copy(file_path, dst_path + "/csv")

    # 遍历old_base_dir_path_unlabeled中的所有csv文件，当其完整路径不在file_path_top_csv中时，将其复制到指定文件夹dst_path_unlabeled
    for file_path in os.listdir(old_base_dir_path_unlabeled + "/csv"):
        if old_base_dir_path_unlabeled + "/csv/" + file_path not in file_path_top_csv:
            shutil.copy(old_base_dir_path_unlabeled + "/csv/" + file_path, dst_path_unlabeled + "/csv")
    
    # 将old_base_dir_path中的rgb和csv文件追加到指定文件夹
    for file_path in os.listdir(old_base_dir_path + "/rgb"):
        shutil.copy(old_base_dir_path + "/rgb/" + file_path, dst_path + "/rgb")
    for file_path in os.listdir(old_base_dir_path + "/csv"):
        shutil.copy(old_base_dir_path + "/csv/" + file_path, dst_path + "/csv")
    
    # 输出各个文件夹中的文件数量
    print("dst_path: ", len(os.listdir(dst_path + "/rgb")))
    print("dst_path_unlabeled: ", len(os.listdir(dst_path_unlabeled + "/rgb")))
    print("old_base_dir_path: ", len(os.listdir(old_base_dir_path + "/rgb")))
    print("old_base_dir_path_unlabeled: ", len(os.listdir(old_base_dir_path_unlabeled + "/rgb")))

    pass

def top_50_file_path(pr_score, file_path):
    # 找到pr_score_final最大的300个图像的file_path
    pr_score_final = []
    for file in file_path:
        pr_score_final.append(pr_score[file]["pr_score_final"])
    pr_score_final = torch.tensor(pr_score_final)
    pr_score_final_top_50 = torch.topk(pr_score_final, 300)
    file_path_top_50 = []
    for i in pr_score_final_top_50[1]:
        file_path_top_50.append(file_path[i])
    return file_path_top_50
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



def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_label_mark", type=str, help="datafile label mark")
    parser.add_argument("--label_U", type=str, help="label_U")
    arg = parser.parse_args()
    print("datafile_label_mark: ", arg.datafile_label_mark)
    print("label_U: ", arg.label_U)
    datafile_label_mark_copy = arg.datafile_label_mark
    label_U_copy = arg.label_U

    data_json_file_path = "./output_double/U2/U2_cor/U2_6/evalu_40_with_298_model/features_map/features_map.json"
    data_coreset_json_file_path = "./output_double/U2/U2_cor/U2_6/evalu_cor_298_with_298_model/features_map/features_map.json"
    old_base_dir_path = "../data_double/U2/data_cor/coco_labeled_298"
    old_base_dir_path_unlabeled = "../data_double/U2/data_cor/coco_unlabeled_40"
    new_rgb_csv_path = "../data_double/U2/data_cor/coco_labeled_318"
    new_rgb_csv_path_unlabeled = "../data_double/U2/data_cor/coco_unlabeled_20"

    if datafile_label_mark_copy == "18_68":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_320_with_18_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_18_with_18_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_18")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_320")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_68")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_270")
    elif datafile_label_mark_copy == "68_118":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_270_with_68_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_68_with_68_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_68")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_270")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_118")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_220")
    elif datafile_label_mark_copy == "118_168":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_220_with_118_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_118_with_118_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_118")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_220")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_168")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_170")
    elif datafile_label_mark_copy == "168_218":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_170_with_168_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_168_with_168_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_168")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_170")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_218")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_120")
    elif datafile_label_mark_copy == "218_268":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_120_with_218_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_218_with_218_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_218")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_120")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_268")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_70")
    elif datafile_label_mark_copy == "268_318":
        data_json_file_path = data_json_file_path.replace("/evalu_40_with_298_model/", "/evalu_70_with_268_model/")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("/evalu_cor_298_with_298_model/", "/evalu_cor_268_with_268_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_298", "/coco_labeled_268")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_40", "/coco_unlabeled_70")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_318", "/coco_labeled_318")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_20", "/coco_unlabeled_20")
    else:
        print("datafile_label_mark error!")
    
    if label_U_copy == "U2":
        pass
    elif label_U_copy == "U3":
        # 将U2_6替换为U3_5
        # 将U2替换为U3
        data_json_file_path = data_json_file_path.replace("U2_6", "U3_5")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("U2_6", "U3_5")
        data_json_file_path = data_json_file_path.replace("U2", "U3")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("U2", "U3")
        old_base_dir_path = old_base_dir_path.replace("U2", "U3")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("U2", "U3")
        new_rgb_csv_path = new_rgb_csv_path.replace("U2", "U3")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("U2", "U3")
    elif label_U_copy == "U4":
        # 将U2_6替换为U4_4
        # 将U2替换为U4
        data_json_file_path = data_json_file_path.replace("U2_6", "U4_4")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("U2_6", "U4_4")
        data_json_file_path = data_json_file_path.replace("U2", "U4")
        data_coreset_json_file_path = data_coreset_json_file_path.replace("U2", "U4")
        old_base_dir_path = old_base_dir_path.replace("U2", "U4")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("U2", "U4")
        new_rgb_csv_path = new_rgb_csv_path.replace("U2", "U4")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("U2", "U4")
    else:
        print("label_U error!")


    data_json = load_json(data_json_file_path)
    data_coreset_json = load_json(data_coreset_json_file_path)

    print("data_json: ", data_json_file_path)
    print("data_coreset_json: ", data_coreset_json_file_path)

    # 查看data_json中"file_path"的值有多少种
    file_path = []
    for item in data_json:
        file_path.append(item["file_path"])
    file_path = list(set(file_path))
    print("file_path: ", len(file_path))
    
    coreset_features_map = []
    for item in data_coreset_json:
        # reshape
        coreset_features_map.append(random_crop(np.array(item["features_map"])))
    print("coreset_features_map: ", len(coreset_features_map))

    # 计算pr_score
    pr_score = get_pr_score(file_path, data_json, coreset_features_map)

    # 找到pr_score_final最大的20个图像的file_path
    file_path_top_50 = top_50_file_path(pr_score, file_path)

    # 将对应位置的图像复制到指定文件夹
    file_copy(file_path_top_50, new_rgb_csv_path, old_base_dir_path, new_rgb_csv_path_unlabeled, old_base_dir_path_unlabeled)

    # 创建coco格式的json文件
    if label_U_copy == "U2":
        csvs_to_coco_6(new_rgb_csv_path)
        csvs_to_coco_6(new_rgb_csv_path_unlabeled)
    elif label_U_copy == "U3":
        csvs_to_coco_5(new_rgb_csv_path)
        csvs_to_coco_5(new_rgb_csv_path_unlabeled)
    elif label_U_copy == "U4":
        csvs_to_coco_4(new_rgb_csv_path)
        csvs_to_coco_4(new_rgb_csv_path_unlabeled)
    pass

if __name__ == "__main__":
    # 尝试运行main函数，如果失败，则一分钟后运行main函数
    try:
        main()
    except:
        print("main function running error")
        import time
        time.sleep(200)
        main()