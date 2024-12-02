# # U3

# # all_three
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_320_with_18_model/evalu_unlab_01.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "all_three" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_68_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_68_80/train_label_01.yaml



# # opu
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_320_with_18_model/evalu_unlab_01.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "opu" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_68_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_68_80/train_label_01.yaml



# # no_p
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "no_p" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_68_80/train_label.yaml



# # unc
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "unc" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_68_80/train_label.yaml



# # cen
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "cen" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_68_80/train_label.yaml



# # ran
# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "ran" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_68_80/train_label.yaml





































# U3


# all_three

# # base train 338 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_bases_338_80/train_bases.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_bases_338_80/train_bases_01.yaml
# # base train 18 abd test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_18_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_18_80/train_label_01.yaml


# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_320_with_18_model/evalu_unlab_01.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "all_three" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_68_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_68_80/train_label_01.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_270_with_68_model/evalu_unlab_01.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "all_three" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_118_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_118_80/train_label_01.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_220_with_118_model/evalu_unlab_01.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "all_three" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_168_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_168_80/train_label_01.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_170_with_168_model/evalu_unlab_01.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "all_three" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_218_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_218_80/train_label_01.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_120_with_218_model/evalu_unlab_01.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "all_three" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_268_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_268_80/train_label_01.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_all_three/my_coco_evalu_70_with_268_model/evalu_unlab_01.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "all_three" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_all_three/my_coco_clabe_318_80/train_label.yaml



































# # U3


# # opu

# # # base train 338 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_bases_338_80/train_bases.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_bases_338_80/train_bases_01.yaml
# # # base train 18 abd test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_18_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_18_80/train_label_01.yaml


# # # evaluate 320 with 18 model
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_320_with_18_model/evalu_unlab_01.yaml
# # # data calculate and add 18 to 68
# # python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "opu" --label_U "U3"
# # # label train 68 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_68_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_68_80/train_label_01.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_270_with_68_model/evalu_unlab_01.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "opu" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_118_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_118_80/train_label_01.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_220_with_118_model/evalu_unlab_01.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "opu" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_168_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_168_80/train_label_01.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_170_with_168_model/evalu_unlab_01.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "opu" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_218_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_218_80/train_label_01.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_120_with_218_model/evalu_unlab_01.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "opu" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_268_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_268_80/train_label_01.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_opu/my_coco_evalu_70_with_268_model/evalu_unlab_01.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "opu" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_opu/my_coco_clabe_318_80/train_label.yaml



































# # U3


# # no_p

# # # base train 338 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_bases_338_80/train_bases.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_bases_338_80/train_bases_01.yaml
# # # base train 18 abd test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_18_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_18_80/train_label_01.yaml


# # # evaluate 320 with 18 model
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # # data calculate and add 18 to 68
# # python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "no_p" --label_U "U3"
# # # label train 68 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_68_80/train_label.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "no_p" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_118_80/train_label.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "no_p" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_168_80/train_label.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "no_p" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_218_80/train_label.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "no_p" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_268_80/train_label.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_no_p/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "no_p" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_no_p/my_coco_clabe_318_80/train_label.yaml














































# # U3


# # unc

# # # base train 338 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_bases_338_80/train_bases.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_bases_338_80/train_bases_01.yaml
# # # base train 18 abd test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_18_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_18_80/train_label_01.yaml


# # # evaluate 320 with 18 model
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # # data calculate and add 18 to 68
# # python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "unc" --label_U "U3"
# # # label train 68 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_68_80/train_label.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "unc" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_118_80/train_label.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "unc" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_168_80/train_label.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "unc" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_218_80/train_label.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "unc" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_268_80/train_label.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_unc/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "unc" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_unc/my_coco_clabe_318_80/train_label.yaml














































# # U3


# # cen

# # # base train 338 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_bases_338_80/train_bases.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_bases_338_80/train_bases_01.yaml
# # # base train 18 abd test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_18_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_18_80/train_label_01.yaml


# # # evaluate 320 with 18 model
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # # data calculate and add 18 to 68
# # python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "cen" --label_U "U3"
# # # label train 68 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_68_80/train_label.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "cen" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_118_80/train_label.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "cen" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_168_80/train_label.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "cen" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_218_80/train_label.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "cen" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_268_80/train_label.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cen/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "cen" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cen/my_coco_clabe_318_80/train_label.yaml














































# U3


# ran

# # base train 338 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_bases_338_80/train_bases.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_bases_338_80/train_bases_01.yaml
# # base train 18 abd test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_18_80/train_label.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_18_80/train_label_01.yaml


# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double/read_pt.py --datafile_label_mark "18_68" --part_mark "ran" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_68_80/train_label.yaml

# evaluate 270 with 68 model
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# data calculate and add 68 to 118
python data_cal/data_cal_double/read_pt.py --datafile_label_mark "68_118" --part_mark "ran" --label_U "U3"
# label train 118 and test 80
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_118_80/train_label.yaml

# evaluate 220 with 118 model
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# data calculate and add 118 to 168
python data_cal/data_cal_double/read_pt.py --datafile_label_mark "118_168" --part_mark "ran" --label_U "U3"
# label train 168 and test 80
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_168_80/train_label.yaml

# evaluate 170 with 168 model
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# data calculate and add 168 to 218
python data_cal/data_cal_double/read_pt.py --datafile_label_mark "168_218" --part_mark "ran" --label_U "U3"
# label train 218 and test 80
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_218_80/train_label.yaml

# evaluate 120 with 218 model
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# data calculate and add 218 to 268
python data_cal/data_cal_double/read_pt.py --datafile_label_mark "218_268" --part_mark "ran" --label_U "U3"
# label train 268 and test 80
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_268_80/train_label.yaml

# evaluate 70 with 268 model
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_ran/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# data calculate and add 268 to 318
python data_cal/data_cal_double/read_pt.py --datafile_label_mark "268_318" --part_mark "ran" --label_U "U3"
# label train 318 and test 80
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_ran/my_coco_clabe_318_80/train_label.yaml














































# # U3


# # cor

# # # base train 338 and test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_bases_338_80/train_bases.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_bases_338_80/train_bases_01.yaml
# # # base train 18 abd test 80
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_18_80/train_label.yaml
# # CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_18_80/train_label_01.yaml


# # evaluate 320 with 18 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_320_with_18_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_18_with_18_model/evalu_unlab.yaml
# # data calculate and add 18 to 68
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "18_68" --label_U "U3"
# # label train 68 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_68_80/train_label.yaml

# # evaluate 270 with 68 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_270_with_68_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_68_with_68_model/evalu_unlab.yaml
# # data calculate and add 68 to 118
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "68_118" --label_U "U3"
# # label train 118 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_118_80/train_label.yaml

# # evaluate 220 with 118 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_220_with_118_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_118_with_118_model/evalu_unlab.yaml
# # data calculate and add 118 to 168
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "118_168" --label_U "U3"
# # label train 168 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_168_80/train_label.yaml

# # evaluate 170 with 168 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_170_with_168_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_168_with_168_model/evalu_unlab.yaml
# # data calculate and add 168 to 218
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "168_218" --label_U "U3"
# # label train 218 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_218_80/train_label.yaml

# # evaluate 120 with 218 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_120_with_218_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_218_with_218_model/evalu_unlab.yaml
# # data calculate and add 218 to 268
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "218_268" --label_U "U3"
# # label train 268 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_268_80/train_label.yaml

# # evaluate 70 with 268 model
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_70_with_268_model/evalu_unlab.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --eval-only --config-file ./configs/coco_double/U3/U3_cor/my_coco_evalu_cor_268_with_268_model/evalu_unlab.yaml
# # data calculate and add 268 to 318
# python data_cal/data_cal_double_cor/read_pt.py --datafile_label_mark "268_318" --label_U "U3"
# # label train 318 and test 80
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/coco_double/U3/U3_cor/my_coco_clabe_318_80/train_label.yaml


