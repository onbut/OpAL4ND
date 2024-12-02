import os
import shutil

# 将源文件夹下的data_clear文件夹复制到目标文件夹，新文件夹名将原文件夹名中的'clear'替换为'all_three','cor','oic','opu','ran','ucl','unc','upl'
def copy_folder(base_folder, mark):
    source_folder = os.path.join(base_folder, mark+'_clear')
    target_folder_1 = os.path.join(base_folder, mark+'_all_three')
    target_folder_2 = os.path.join(base_folder, mark+'_cor')
    target_folder_3 = os.path.join(base_folder, mark+'_oic')
    target_folder_4 = os.path.join(base_folder, mark+'_opu')
    target_folder_5 = os.path.join(base_folder, mark+'_ran')
    target_folder_6 = os.path.join(base_folder, mark+'_ucl')
    target_folder_7 = os.path.join(base_folder, mark+'_unc')
    target_folder_8 = os.path.join(base_folder, mark+'_upl')

    target_folder_list = [target_folder_1, target_folder_2, target_folder_3, target_folder_4, target_folder_5, target_folder_6, target_folder_7, target_folder_8]

    for target_folder in target_folder_list:
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        # 对整个文件夹所有内容进行复制
        print('cp -r ' + source_folder + '/* ' + target_folder)
        os.system('cp -r ' + source_folder + '/* ' + target_folder)
        print('cp -r ' + source_folder + '/* ' + target_folder + ' done')
        pass

    pass

def copy_big_folder(base_folder_1, base_folder_2):
    source_folder_1 = os.path.join(base_folder_1, 'data_double')
    target_folder_1 = os.path.join(base_folder_1, 'data_double_2')
    source_folder_2 = os.path.join(base_folder_2, 'output_double')
    target_folder_2 = os.path.join(base_folder_2, 'output_double_2')

    if not os.path.exists(target_folder_1):
        os.makedirs(target_folder_1)
    if not os.path.exists(target_folder_2):
        os.makedirs(target_folder_2)

    # 对整个文件夹所有内容进行复制
    print('cp -r ' + source_folder_1 + '/* ' + target_folder_1)
    os.system('cp -r ' + source_folder_1 + '/* ' + target_folder_1)
    print('cp -r ' + source_folder_1 + '/* ' + target_folder_1 + ' done')

    print('cp -r ' + source_folder_2 + '/* ' + target_folder_2)
    os.system('cp -r ' + source_folder_2 + '/* ' + target_folder_2)
    print('cp -r ' + source_folder_2 + '/* ' + target_folder_2 + ' done')
    pass

def copy_3_times(base_folder, mark):
    source_folder = os.path.join(base_folder, mark+'_base')
    target_folder = os.path.join(base_folder, mark+'_double')

    target_folder_1 = target_folder + '_1'
    target_folder_2 = target_folder + '_2'
    target_folder_3 = target_folder + '_3'

    if not os.path.exists(target_folder_1):
        os.makedirs(target_folder_1, exist_ok=True)
    if not os.path.exists(target_folder_2):
        os.makedirs(target_folder_2, exist_ok=True)
    if not os.path.exists(target_folder_3):
        os.makedirs(target_folder_3, exist_ok=True)

    # 对整个文件夹所有内容进行复制
    print('cp -r ' + source_folder + '/* ' + target_folder_1)
    os.system('cp -r ' + source_folder + '/* ' + target_folder_1)
    print('cp -r ' + source_folder + '/* ' + target_folder_1 + ' done')

    print('cp -r ' + source_folder + '/* ' + target_folder_2)
    os.system('cp -r ' + source_folder + '/* ' + target_folder_2)
    print('cp -r ' + source_folder + '/* ' + target_folder_2 + ' done')

    print('cp -r ' + source_folder + '/* ' + target_folder_3)
    os.system('cp -r ' + source_folder + '/* ' + target_folder_3)
    print('cp -r ' + source_folder + '/* ' + target_folder_3 + ' done')
    pass


def test_cuda():
    import torch
    print(torch.cuda.is_available())

def main():
    # copy_folder('../data_base/U2', 'data')
    # copy_folder('../data_base/U3', 'data')
    # copy_folder('../data_base/U4', 'data')

    # copy_3_times('..', 'data')

    # copy_folder('./output_double/U2', 'U2')
    # copy_folder('./output_double/U3', 'U3')
    # copy_folder('./output_double/U4', 'U4')

    copy_3_times('.', 'output')

    # copy_big_folder('..', '.')
    # test_cuda()

    pass

if __name__ == '__main__':
    main()
    pass