import os
import argparse
import torch
import json
import numpy as np

# U2 - U2_all_three - U2_01/U2_6 - clabe_18_80/clabe_68_80/clabe_118_80/clabe_168_80/clabe_218_80/clabe_268_80/clabe_318_80/base_338_80 - ab_save - clas_mean - 12000.pt
# U3 - U3_all_three - U3_01/U3_5 - clabe_18_80/clabe_68_80/clabe_118_80/clabe_168_80/clabe_218_80/clabe_268_80/clabe_318_80/base_338_80 - ab_save - clas_mean - 12000.pt
# U4 - U4_all_three - U4_01/U4_4 - clabe_18_80/clabe_68_80/clabe_118_80/clabe_168_80/clabe_218_80/clabe_268_80/clabe_318_80/base_338_80 - ab_save - clas_mean - 12000.pt

def get_all_centers(tar_path, dir_path):
    # 文件夹组织如上，我要创建一个字典文件，存储每一个文件的路径, keys: clabe_18_80, clabe_68_80, clabe_118_80, clabe_168_80, clabe_218_80, clabe_268_80, clabe_318_80, base_338_80
    # values也是一个字典，存储每一个文件下的所有文件路径，keys:U2_01, U2_6, U3_01, U3_5, U4_01, U4_4

    base_dir_for_U2 = os.path.join(dir_path, 'U2', 'U2_all_three')
    base_dir_for_U3 = os.path.join(dir_path, 'U3', 'U3_all_three')
    base_dir_for_U4 = os.path.join(dir_path, 'U4', 'U4_all_three')

    sub_keys_list = ['U2_01', 'U3_01', 'U4_01']
    keys_list = ['clabe_18_80', 'clabe_68_80', 'clabe_118_80', 'clabe_168_80', 'clabe_218_80', 'clabe_268_80', 'clabe_318_80', 'bases_338_80']

    pt_file_path = os.path.join('ab_save', 'cls_mean', '12000.pt')
    feat_file_path = os.path.join('inference_store', 'feat.pt')

    # lizard
    # names_list_U2_6 = ['tumor', 'lymphoc', 'stromal', 'plasma_f', 'mitotic_f', 'background']
    # names_list_U3_5 = ['tumor', 'lymphoc', 'stromal', 'plasma_f', 'background']
    # names_list_U4_4 = ['tumor', 'lymphoc', 'stromal', 'background']
    # names_list_U2_6 = ['tumor', 'lymphoc', 'stromal', 'plasma_f', 'mitotic_f']
    # names_list_U3_5 = ['tumor', 'lymphoc', 'stromal', 'plasma_f']
    # names_list_U4_4 = ['tumor', 'lymphoc', 'stromal']
    # panuke
    names_list_U2_6 = ['Neoplastic', 'Stromal', 'Inflammatory', 'Dead']
    names_list_U3_5 = ['Neoplastic', 'Stromal', 'Inflammatory']
    names_list_U4_4 = ['Neoplastic', 'Stromal']
    names_list_UX_01 = ['background', 'known', 'unknown']

    count = 0

    centers = {}
    features = {}

    for key in keys_list:
        centers[key] = {}
        features[key] = {}
        for sub_key in sub_keys_list:
            centers[key][sub_key] = {}
            features[key][sub_key] = {}
            if 'U2' in sub_key:
                use_file_path = os.path.join(base_dir_for_U2, sub_key, key, pt_file_path)
                use_feat_file_path = os.path.join(base_dir_for_U2, sub_key, key, feat_file_path)
            elif 'U3' in sub_key:
                use_file_path = os.path.join(base_dir_for_U3, sub_key, key, pt_file_path)
                use_feat_file_path = os.path.join(base_dir_for_U3, sub_key, key, feat_file_path)
            elif 'U4' in sub_key:
                use_file_path = os.path.join(base_dir_for_U4, sub_key, key, pt_file_path)
                use_feat_file_path = os.path.join(base_dir_for_U4, sub_key, key, feat_file_path)
            else:
                raise ValueError('sub_key error')
            
            if '_01' in sub_key and 'clabe_318_80' in key:
                # 跳过
                continue
            
            # if 'U4' in sub_key:
            #     # 跳过
            #     continue
            
            if 'U2_6' in sub_key:
                names_list = names_list_U2_6
            elif 'U3_5' in sub_key:
                names_list = names_list_U3_5
            elif 'U4_4' in sub_key:
                names_list = names_list_U4_4
            elif '_01' in sub_key:
                names_list = names_list_UX_01
            else:
                raise ValueError('sub_key error')
            
            # 读取.pt文件
            pt_centers = torch.load(use_file_path, weights_only=True)
            pt_features = torch.load(use_feat_file_path)
            # 查看pt_centers的形状
            print('pt_centers shape:', len(pt_centers))    
            print(len(names_list))
            if len(pt_centers) != len(names_list)+1:
                print("use_file_path:", use_file_path)
                # raise ValueError('pt_centers shape error')
            for i, name in enumerate(names_list):
                centers[key][sub_key][name] = pt_centers[i].tolist()
                curr_list = []
                class_feature_deque = pt_features.store[i]
                for sample_feature in class_feature_deque:
                    curr_list.append(sample_feature.tolist())
                features[key][sub_key][name] = curr_list
                # print('curr_list:', len(curr_list))

            count += 1
            print('count:', count)
    
    # 保存字典
    all_info = {'centers': centers, 'features': features}
    with open(tar_path, 'w') as f:
        json.dump(all_info, f)

    pass
            
def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_path', type=str, default='cnenter_dir.json') # ap or cell
    
    parser.add_argument('--dir_path', type=str, default='output_double')

    args = parser.parse_args()
    tar_path = args.tar_path
    dir_path = args.dir_path
    print('apd_mark:', tar_path)
    print('dir_path:', dir_path)

    get_all_centers(tar_path, dir_path)

    # all_ap(dir_path)
    # all_for_U(dir_path)
    
    pass

if __name__ == '__main__':
    main()
    # dir_root = '/home/yyg/workspace/openset/all/data_double/U3'
    # print('dir_root:', dir_root)
    # all_for_U(dir_root)

    # dir_root = '/home/yyg/workspace/openset/all/data_double/U2'
    # print('dir_root:', dir_root)
    # all_for_U(dir_root)
    pass


