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
from sklearn.cluster import KMeans
from json import dump
from json import load as jsload


def load_pt_file(file_path):
    data = torch.load(file_path)
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = jsload(f)
    return data

def purity_cal(input_features_save, cls_mean):
    # 将输入转换为torch张量
    input_features_save = torch.tensor(input_features_save)
    cos_sim = []
    for cls_mean_item in cls_mean:
        cls_mean_item = cls_mean_item.clone().detach()
        # 计算余弦相似度
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim.append(cos(input_features_save, cls_mean_item))
    max_cos_sim = torch.max(torch.tensor(cos_sim))
    # 当cos_sim最后一项为最大值时，返回标量0，否则返回标量1
    if cos_sim.index(max_cos_sim) == len(cos_sim)-1:
        return 0
    else:
        return 1

def info_cls_cal(cls_sc_save):
    # 将输入转换为torch张量
    eb_save = torch.tensor(cls_sc_save)

    # 为cls_sc_save每一个元素加上一个1，获得alpha
    alpha = eb_save + 1

    # 计算Dirichlet分布的熵
    alpha_sum = torch.sum(alpha)
    
    # 信念质量
    belief = eb_save / alpha_sum
    return 1-sum(belief).item()  # 返回Python标量

def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def info_loc_cal(bbox, proposal_bbox):
    # 将输入转换为torch张量
    bbox = torch.tensor(bbox)
    proposal_bbox = torch.tensor(proposal_bbox)

    # 计算IOU
    iou = bbox_iou(bbox, proposal_bbox)

    return 1-iou  # 返回Python标量

def uncertainty_cal(cls_scores):
    # 将输入转换为torch张量
    cls_scores = torch.tensor(cls_scores)

    # 计算每个类别的预测概率
    probs = torch.softmax(cls_scores, dim=0)

    # 计算熵
    entropy = -torch.sum(probs * torch.log(probs))

    return entropy.item()  # 返回Python标量


def get_cluster_score(file, data_json):
    # 将file_path为file的所有图像的input_features_save和category_id提取出来
    data_use_for_file = []
    input_features_save = []
    category_id = []

    for item in data_json:
        if item["file_path"] == file:
            data_use_for_file.append(item)
            input_features_save.append(item["input_features_save"])
            category_id.append(item["category_id"])
            pass
    
    # 计算category_id的种类数
    category_id = list(set(category_id))
    category_id_num = len(category_id)

    # 进行KMeans聚类
    kmeans = KMeans(n_clusters=category_id_num, random_state=0).fit(input_features_save)
    
    # 计算每个样例到达其对应聚类中心的距离
    distance = kmeans.transform(input_features_save)

    # 对距离归一化
    distance = distance / (distance + 1)

    # 计算均值
    mean_distance = torch.mean(torch.tensor(distance))

    return mean_distance.item()


def get_pr_score(file_path, data_json, data_cls_mean, data_json_cls_mean):
    pr_score = {}
    for file in file_path:
        pr_score[file] = {}
        
        pr_score_purity = []
        pr_score_info_cls = []
        pr_score_info_loc = []
        pr_score_uncertainty = []

        for item in data_json_cls_mean:
            if item["file_path"] == file:
                pr_score_purity.append(purity_cal(item["input_features_save"], data_cls_mean))
                pass
        
        for item in data_json:
            if item["file_path"] == file:
                pr_score_info_cls.append(info_cls_cal(item["cls_sc_save"]))
                pr_score_info_loc.append(info_loc_cal(item["bbox"], item["proposal_bbox"]))
                pr_score_uncertainty.append(uncertainty_cal(item["cls_sc_save"]))
                pass

        pr_score[file]["putity_score"] = sum(pr_score_purity)
        pr_score[file]["info_cls_score"] = sum(pr_score_info_cls) / len(pr_score_info_cls)
        pr_score[file]["info_loc_score"] = sum(pr_score_info_loc) / len(pr_score_info_loc)
        pr_score[file]["uncertainty_score"] = sum(pr_score_uncertainty) / len(pr_score_uncertainty)

        # uncer_score = pr_score[file]["info_cls_score"] * pr_score[file]["info_loc_score"]
        pr_score[file]["uncer_score"] = pr_score[file]["info_cls_score"] * pr_score[file]["info_loc_score"]
        # cluster_score
        pr_score[file]["cluster_score"] = get_cluster_score(file, data_json)
        # pr_v2_score
        pr_score[file]["pr_v2_score"] = pr_score[file]["uncer_score"] + pr_score[file]["cluster_score"]
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

def nolmalization(pr_score, file_path, part_mark_copy):
    # 所有图像的purity_score, info_cls_score, info_loc_score, uncertainty_score归一化
    purity_score = []
    info_cls_score = []
    info_loc_score = []
    uncertainty_score = []
    for file in file_path:
        purity_score.append(pr_score[file]["putity_score"])
        info_cls_score.append(pr_score[file]["info_cls_score"])
        info_loc_score.append(pr_score[file]["info_loc_score"])
        uncertainty_score.append(pr_score[file]["uncertainty_score"])
    purity_score = torch.tensor(purity_score)
    info_cls_score = torch.tensor(info_cls_score)
    info_loc_score = torch.tensor(info_loc_score)
    uncertainty_score = torch.tensor(uncertainty_score)
    purity_score = (purity_score - torch.min(purity_score)) / (torch.max(purity_score) - torch.min(purity_score))
    info_cls_score = (info_cls_score - torch.min(info_cls_score)) / (torch.max(info_cls_score) - torch.min(info_cls_score))
    info_loc_score = (info_loc_score - torch.min(info_loc_score)) / (torch.max(info_loc_score) - torch.min(info_loc_score))
    uncertainty_score = (uncertainty_score - torch.min(uncertainty_score)) / (torch.max(uncertainty_score) - torch.min(uncertainty_score))
    for file in file_path:
        pr_score[file]["putity_score"] = purity_score[file_path.index(file)].item()
        pr_score[file]["info_cls_score"] = info_cls_score[file_path.index(file)].item()
        pr_score[file]["info_loc_score"] = info_loc_score[file_path.index(file)].item()
        pr_score[file]["info_score"] = pr_score[file]["info_cls_score"] * pr_score[file]["info_loc_score"]
        pr_score[file]["uncertainty_score_for_info"] = uncertainty_score[file_path.index(file)].item()
        pr_score[file]["info_uncertainty_loc_score"] = pr_score[file]["info_loc_score"] * pr_score[file]["uncertainty_score_for_info"]
        pr_score[file]["uncertainty_plus_info_loc_score"] = pr_score[file]["info_loc_score"] + pr_score[file]["uncertainty_score_for_info"]
        # if part_mark_copy == "all_three":
        #     pr_score[file]["pr_score_final"] = pr_score[file]["putity_score"] * pr_score[file]["info_cls_score"] * pr_score[file]["info_loc_score"]
        # elif part_mark_copy == "no_p":
        #     pr_score[file]["pr_score_final"] = pr_score[file]["info_cls_score"] * pr_score[file]["info_loc_score"]
        # elif part_mark_copy == "opu":
        #     pr_score[file]["pr_score_final"] = pr_score[file]["putity_score"]
        # elif part_mark_copy == "ran":
        #     # 随机生成一个0-1之间的数
        #     pr_score[file]["pr_score_final"] = torch.rand(1).item()
    return pr_score


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

def top_100_purity_file_path(pr_score, file_path):
    # 找到putity_score最大的60个图像的file_path
    putity_score = []
    for file in file_path:
        putity_score.append(pr_score[file]["putity_score"])
    putity_score = torch.tensor(putity_score)
    if len(putity_score) < 60:
        return file_path
    putity_score_top_100 = torch.topk(putity_score, 60)
    file_path_top_100 = []
    for i in putity_score_top_100[1]:
        file_path_top_100.append(file_path[i])
    return file_path_top_100

def top_50_info_file_path(pr_score, file_path):
    # 找到info_score最大的300个图像的file_path
    info_score = []
    for file in file_path:
        info_score.append(pr_score[file]["info_score"])
    info_score = torch.tensor(info_score)
    info_score_top_50 = torch.topk(info_score, 300)
    file_path_top_50 = []
    for i in info_score_top_50[1]:
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


def csvs_to_coco_01(dor_path):
    image_dir = os.path.join(dor_path, 'rgb')
    csv_dir = os.path.join(dor_path, 'csv')
    output_directory = os.path.join(dor_path, 'json')
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


def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_label_mark", type=str, help="datafile label mark")
    parser.add_argument("--part_mark", type=str, help="part mark")
    parser.add_argument("--label_U", type=str, help="label U")
    arg = parser.parse_args()
    print("datafile_label_mark: ", arg.datafile_label_mark)
    print("part_mark: ", arg.part_mark)
    print("label_U: ", arg.label_U)
    part_mark_copy = arg.part_mark
    datafile_label_mark_copy = arg.datafile_label_mark
    label_U_copy = arg.label_U

    cls_mean_file_path = "./output_double/U2/U2_all_three/U2_01/clabe_138_80/ab_save/cls_mean/12000.pt"
    data_json_file_path = "./output_double/U2/U2_all_three/U2_6/evalu_200_with_138_model/inference/coco_instances_results.json"
    old_base_dir_path = "../data_double/U2/data_all_three/coco_labeled_138"
    old_base_dir_path_unlabeled = "../data_double/U2/data_all_three/coco_unlabeled_200"
    new_rgb_csv_path = "../data_double/U2/data_all_three/coco_labeled_158"
    new_rgb_csv_path_unlabeled = "../data_double/U2/data_all_three/coco_unlabeled_180"

    if part_mark_copy == "all_three":
        # 不做任何处理
        pass
    elif part_mark_copy == "ucl":
        # 将U2_all_three替换为U2_ucl
        # 将data_all_three替换为data_ucl
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_ucl/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_ucl/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_ucl/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_ucl/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_ucl/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_ucl/")
    elif part_mark_copy == "oic":
        # 将U2_all_three替换为U2_oic
        # 将data_all_three替换为data_oic
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_oic/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_oic/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_oic/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_oic/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_oic/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_oic/")
    elif part_mark_copy == "upl":
        # 将U2_all_three替换为U2_upl
        # 将data_all_three替换为data_upl
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_upl/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_upl/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_upl/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_upl/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_upl/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_upl/")
    elif part_mark_copy == "no_ic":
        # 将U2_all_three替换为U2_no_ic
        # 将data_all_three替换为data_no_ic
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_no_ic/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_no_ic/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_no_ic/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_no_ic/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_no_ic/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_no_ic/")
    elif part_mark_copy == "no_il":
        # 将U2_all_three替换为U2_no_il
        # 将data_all_three替换为data_no_il
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_no_il/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_no_il/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_no_il/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_no_il/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_no_il/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_no_il/")
    elif part_mark_copy == "no_p":
        # 将U2_all_three替换为U2_no_p
        # 将data_all_three替换为data_no_p
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_no_p/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_no_p/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_no_p/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_no_p/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_no_p/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_no_p/")
    elif part_mark_copy == "opu":
        # 将U2_all_three替换为U2_opu
        # 将data_all_three替换为data_opu
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_opu/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_opu/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_opu/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_opu/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_opu/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_opu/")
    elif part_mark_copy == "cor":
        # 将U2_all_three替换为U2_cor
        # 将data_all_three替换为data_cor
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_cor/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_cor/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_cor/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_cor/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_cor/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_cor/")
    elif part_mark_copy == "ran":
        # 将U2_all_three替换为U2_ran
        # 将data_all_three替换为data_ran
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_ran/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_ran/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_ran/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_ran/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_ran/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_ran/")
    elif part_mark_copy == "unc":
        # 将U2_all_three替换为U2_unc
        # 将data_all_three替换为data_unc
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_unc/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_unc/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_unc/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_unc/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_unc/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_unc/")
    elif part_mark_copy == "cen":
        # 将U2_all_three替换为U2_cen
        # 将data_all_three替换为data_cen
        cls_mean_file_path = cls_mean_file_path.replace("/U2_all_three/", "/U2_cen/")
        data_json_file_path = data_json_file_path.replace("/U2_all_three/", "/U2_cen/")
        old_base_dir_path = old_base_dir_path.replace("/data_all_three/", "/data_cen/")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/data_all_three/", "/data_cen/")
        new_rgb_csv_path = new_rgb_csv_path.replace("/data_all_three/", "/data_cen/")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/data_all_three/", "/data_cen/")
    else:
        print("part_mark error!")

    if part_mark_copy == "ran":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    elif part_mark_copy == "unc":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    elif part_mark_copy == "cen":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    elif part_mark_copy == "no_p":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    elif part_mark_copy == "oic":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    elif part_mark_copy == "upl":
        cls_mean_file_path = cls_mean_file_path.replace("/U2_01/", "/U2_6/")
    
    if datafile_label_mark_copy == "18_68":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_18_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_320_with_18_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_18")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_320")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_68")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_270")
    elif datafile_label_mark_copy == "68_118":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_68_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_270_with_68_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_68")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_270")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_118")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_220")
    elif datafile_label_mark_copy == "118_168":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_118_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_220_with_118_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_118")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_220")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_168")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_170")
    elif datafile_label_mark_copy == "168_218":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_168_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_170_with_168_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_168")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_170")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_218")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_120")
    elif datafile_label_mark_copy == "218_268":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_218_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_120_with_218_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_218")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_120")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_268")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_70")
    elif datafile_label_mark_copy == "268_318":
        cls_mean_file_path = cls_mean_file_path.replace("/clabe_138_80/", "/clabe_268_80/")
        data_json_file_path = data_json_file_path.replace("/evalu_200_with_138_model/", "/evalu_70_with_268_model/")
        old_base_dir_path = old_base_dir_path.replace("/coco_labeled_138", "/coco_labeled_268")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("/coco_unlabeled_200", "/coco_unlabeled_70")
        new_rgb_csv_path = new_rgb_csv_path.replace("/coco_labeled_158", "/coco_labeled_318")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("/coco_unlabeled_180", "/coco_unlabeled_20")
    else:
        print("datafile_label_mark error!")
        

    data_json_file_path_cls_mean = data_json_file_path.replace("U2_6", "U2_01")

    
    if part_mark_copy == "ran":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")
    elif part_mark_copy == "unc":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")
    elif part_mark_copy == "cen":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")
    elif part_mark_copy == "no_p":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")
    elif part_mark_copy == "oic":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")
    elif part_mark_copy == "upl":
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_01", "U2_6")

    if label_U_copy == "U2":
        pass
    elif label_U_copy == "U3":
        # 将U2_6替换为U3_5
        # 将U2替换为U3
        cls_mean_file_path = cls_mean_file_path.replace("U2_6", "U3_5")
        data_json_file_path = data_json_file_path.replace("U2_6", "U3_5")
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_6", "U3_5")
        cls_mean_file_path = cls_mean_file_path.replace("U2", "U3")
        data_json_file_path = data_json_file_path.replace("U2", "U3")
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2", "U3")
        old_base_dir_path = old_base_dir_path.replace("U2", "U3")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("U2", "U3")
        new_rgb_csv_path = new_rgb_csv_path.replace("U2", "U3")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("U2", "U3")
    elif label_U_copy == "U4":
        # 将U2_6替换为U4_4
        # 将U2替换为U4
        cls_mean_file_path = cls_mean_file_path.replace("U2_6", "U4_4")
        data_json_file_path = data_json_file_path.replace("U2_6", "U4_4")
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2_6", "U4_4")
        cls_mean_file_path = cls_mean_file_path.replace("U2", "U4")
        data_json_file_path = data_json_file_path.replace("U2", "U4")
        data_json_file_path_cls_mean = data_json_file_path_cls_mean.replace("U2", "U4")
        old_base_dir_path = old_base_dir_path.replace("U2", "U4")
        old_base_dir_path_unlabeled = old_base_dir_path_unlabeled.replace("U2", "U4")
        new_rgb_csv_path = new_rgb_csv_path.replace("U2", "U4")
        new_rgb_csv_path_unlabeled = new_rgb_csv_path_unlabeled.replace("U2", "U4")
    else:
        print("label_U error!")

        

    data_cls_mean = load_pt_file(cls_mean_file_path)
    data_json = load_json(data_json_file_path)
    data_json_cls_mean = load_json(data_json_file_path_cls_mean)

    print("data_cls_mean: ", cls_mean_file_path)
    print("data_json: ", data_json_file_path)
    print("data_json_cls_mean: ", data_json_file_path_cls_mean)

    # 查看data_json中"file_path"的值有多少种
    file_path = []
    for item in data_json:
        file_path.append(item["file_path"])
    file_path = list(set(file_path))
    print("file_path: ", len(file_path))

    
    # 计算pr_score
    pr_score = get_pr_score(file_path, data_json, data_cls_mean, data_json_cls_mean)

    # 归一化
    pr_score = nolmalization(pr_score, file_path, part_mark_copy)

    file_path_top_50 = []

    if part_mark_copy == "all_three":
        # # 记录purity_score最大的100个图像的file_path
        # file_path_top_100 = top_100_purity_file_path(pr_score, file_path)
        # # 记录file_path_top_100中info_score最大的50个图像的file_path
        # file_path_top_50 = top_50_info_file_path(pr_score, file_path_top_100)
        # 将pr_v2_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["pr_v2_score"]

        # 找到pr_v2_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)

    elif part_mark_copy == "ucl":
        # 记录purity_score最大的100个图像的file_path
        file_path_top_100 = top_100_purity_file_path(pr_score, file_path)
        # 将info_score替换为info_uncertainty_loc_score
        for file in file_path:
            pr_score[file]["info_score"] = pr_score[file]["info_uncertainty_loc_score"]
        # 找到info_uncertainty_loc_score最大的50个图像的file_path
        file_path_top_50 = top_50_info_file_path(pr_score, file_path_top_100)
    elif part_mark_copy == "oic":
        # 将info_cls_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["info_cls_score"]
        # 找到info_cls_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "upl":
        # 将uncertainty_plus_info_loc_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["uncertainty_plus_info_loc_score"]
        # 找到uncertainty_plus_info_loc_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "no_ic":
        file_path_top_100 = top_100_purity_file_path(pr_score, file_path)
        # 将info_loc_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["info_loc_score"]
        # 找到info_loc_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path_top_100)
    elif part_mark_copy == "no_il":
        file_path_top_100 = top_100_purity_file_path(pr_score, file_path)
        # 将info_cls_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["info_cls_score"]
        # 找到info_cls_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path_top_100)
    elif part_mark_copy == "no_p":
        # 将info_socre作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["info_score"]
        # 找到info_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "opu":
        # 将putity_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["putity_score"]
        # 找到putity_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "ran":
        # 随机生成一个0-1之间的数
        for file in file_path:
            pr_score[file]["pr_score_final"] = torch.rand(1).item()
        # 找到随机生成的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "unc":
        # 将uncertainty_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = pr_score[file]["uncertainty_score"]
        # 找到uncertainty_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    elif part_mark_copy == "cen":
        # 将1-uncertainty_score作为pr_score_final
        for file in file_path:
            pr_score[file]["pr_score_final"] = 1 - pr_score[file]["uncertainty_score"]
        # 找到1-uncertainty_score最大的50个图像的file_path
        file_path_top_50 = top_50_file_path(pr_score, file_path)
    else:
        print("part_mark error!")


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
        
    if part_mark_copy == "all_three":
        csvs_to_coco_01(new_rgb_csv_path)
        csvs_to_coco_01(new_rgb_csv_path_unlabeled)
    elif part_mark_copy == "opu":
        csvs_to_coco_01(new_rgb_csv_path)
        csvs_to_coco_01(new_rgb_csv_path_unlabeled)
    elif part_mark_copy == "no_ic":
        csvs_to_coco_01(new_rgb_csv_path)
        csvs_to_coco_01(new_rgb_csv_path_unlabeled)
    elif part_mark_copy == "no_il":
        csvs_to_coco_01(new_rgb_csv_path)
        csvs_to_coco_01(new_rgb_csv_path_unlabeled)
    elif part_mark_copy == "ucl":
        csvs_to_coco_01(new_rgb_csv_path)
        csvs_to_coco_01(new_rgb_csv_path_unlabeled)
    pass

if __name__ == "__main__":
    # 尝试运行main函数，如果失败，则一分钟后运行main函数
    try:
        main()
    except:
        import time
        time.sleep(300)
        main()