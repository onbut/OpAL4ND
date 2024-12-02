#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

# My imports
from detectron2.data.datasets import register_coco_instances

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name, cfg)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# My functions
def my_register_coco_dataset(args):
    cfg_data = get_cfg()
    cfg_data.merge_from_file(args.config_file)

    # 获取数据集的名称
    train_dataset_name = cfg_data.DATASETS.TRAIN[0]
    test_dataset_name = cfg_data.DATASETS.TEST[0]

    base_dir_path = "/home/yagao/workspace/openset/mydete/"
    base_dir_path_ablation = "data/"
    if 'no_p' in test_dataset_name:
        base_dir_path_ablation = "data_no_p/"
    elif 'no_ic' in test_dataset_name:
        base_dir_path_ablation = "data_no_ic/"
    elif 'no_il' in test_dataset_name:
        base_dir_path_ablation = "data_no_il/"
    elif 'ran' in test_dataset_name:
        base_dir_path_ablation = "data_ran/"
        pass

    dataset_mapping_dict = {
        "train_U4_bases_338_4": "U4/coco_base_train_338/",
        "train_U3_bases_338_5": "U3/coco_base_train_338/",
        "train_U2_bases_338_6": "U2/coco_base_train_338/",
        "test_U4_bases_80_4": "U4/coco_base_test_80/",
        "test_U3_bases_80_5": "U3/coco_base_test_80/",
        "test_U2_bases_80_6": "U2/coco_base_test_80/",

        "label_U4_bases_138_4": "U4/coco_labeled_138/",
        "label_U4_label_158_4": "U4/coco_labeled_158/",
        "label_U4_label_178_4": "U4/coco_labeled_178/",
        "label_U4_label_198_4": "U4/coco_labeled_198/",
        "label_U4_label_218_4": "U4/coco_labeled_218/",
        "label_U4_label_238_4": "U4/coco_labeled_238/",
        "label_U4_label_258_4": "U4/coco_labeled_258/",
        "label_U4_label_278_4": "U4/coco_labeled_278/",
        "label_U4_label_298_4": "U4/coco_labeled_298/",
        "label_U4_label_318_4": "U4/coco_labeled_318/",

        "unlab_U4_20_4": "U4/coco_unlabeled_20/",
        "unlab_U4_40_4": "U4/coco_unlabeled_40/",
        "unlab_U4_60_4": "U4/coco_unlabeled_60/",
        "unlab_U4_80_4": "U4/coco_unlabeled_80/",
        "unlab_U4_100_4": "U4/coco_unlabeled_100/",
        "unlab_U4_120_4": "U4/coco_unlabeled_120/",
        "unlab_U4_140_4": "U4/coco_unlabeled_140/",
        "unlab_U4_160_4": "U4/coco_unlabeled_160/",
        "unlab_U4_180_4": "U4/coco_unlabeled_180/",
        "unlab_U4_200_4": "U4/coco_unlabeled_200/",

        "label_U3_bases_138_5": "U3/coco_labeled_138/",
        "label_U3_label_158_5": "U3/coco_labeled_158/",
        "label_U3_label_178_5": "U3/coco_labeled_178/",
        "label_U3_label_198_5": "U3/coco_labeled_198/",
        "label_U3_label_218_5": "U3/coco_labeled_218/",
        "label_U3_label_238_5": "U3/coco_labeled_238/",
        "label_U3_label_258_5": "U3/coco_labeled_258/",
        "label_U3_label_278_5": "U3/coco_labeled_278/",
        "label_U3_label_298_5": "U3/coco_labeled_298/",
        "label_U3_label_318_5": "U3/coco_labeled_318/",

        "unlab_U3_20_5": "U3/coco_unlabeled_20/",
        "unlab_U3_40_5": "U3/coco_unlabeled_40/",
        "unlab_U3_60_5": "U3/coco_unlabeled_60/",
        "unlab_U3_80_5": "U3/coco_unlabeled_80/",
        "unlab_U3_100_5": "U3/coco_unlabeled_100/",
        "unlab_U3_120_5": "U3/coco_unlabeled_120/",
        "unlab_U3_140_5": "U3/coco_unlabeled_140/",
        "unlab_U3_160_5": "U3/coco_unlabeled_160/",
        "unlab_U3_180_5": "U3/coco_unlabeled_180/",
        "unlab_U3_200_5": "U3/coco_unlabeled_200/",

        "label_U2_bases_138_6": "U2/coco_labeled_138/",
        "label_U2_label_158_6": "U2/coco_labeled_158/",
        "label_U2_label_178_6": "U2/coco_labeled_178/",
        "label_U2_label_198_6": "U2/coco_labeled_198/",
        "label_U2_label_218_6": "U2/coco_labeled_218/",
        "label_U2_label_238_6": "U2/coco_labeled_238/",
        "label_U2_label_258_6": "U2/coco_labeled_258/",
        "label_U2_label_278_6": "U2/coco_labeled_278/",
        "label_U2_label_298_6": "U2/coco_labeled_298/",
        "label_U2_label_318_6": "U2/coco_labeled_318/",

        "unlab_U2_20_6": "U2/coco_unlabeled_20/",
        "unlab_U2_40_6": "U2/coco_unlabeled_40/",
        "unlab_U2_60_6": "U2/coco_unlabeled_60/",
        "unlab_U2_80_6": "U2/coco_unlabeled_80/",
        "unlab_U2_100_6": "U2/coco_unlabeled_100/",
        "unlab_U2_120_6": "U2/coco_unlabeled_120/",
        "unlab_U2_140_6": "U2/coco_unlabeled_140/",
        "unlab_U2_160_6": "U2/coco_unlabeled_160/",
        "unlab_U2_180_6": "U2/coco_unlabeled_180/",
        "unlab_U2_200_6": "U2/coco_unlabeled_200/",
    }

    if "U4" in test_dataset_name:
        json_add = "json/coco_4.json"
    elif "U3" in test_dataset_name:
        json_add = "json/coco_5.json"
    elif "U2" in test_dataset_name:
        json_add = "json/coco_6.json"
        pass

    # 新建一个train_new_name，如果原train_dataset_name中包含_no_p_，或者_no_ic_，或者_no_il_，或者_ran_，则将其去掉
    train_new_name = train_dataset_name
    test_new_name = test_dataset_name
    if 'no_p' in test_dataset_name:
        train_new_name = train_dataset_name.replace('no_p_', '')
        test_new_name = test_dataset_name.replace('no_p_', '')
    elif 'no_ic' in test_dataset_name:
        train_new_name = train_dataset_name.replace('no_ic_', '')
        test_new_name = test_dataset_name.replace('no_ic_', '')
    elif 'no_il' in test_dataset_name:
        train_new_name = train_dataset_name.replace('no_il_', '')
        test_new_name = test_dataset_name.replace('no_il_', '')
    elif 'ran' in test_dataset_name:
        train_new_name = train_dataset_name.replace('ran_', '')
        test_new_name = test_dataset_name.replace('ran_', '')
        pass

    if train_new_name in dataset_mapping_dict.keys():
        json_path = base_dir_path + base_dir_path_ablation + dataset_mapping_dict[train_new_name] + json_add
        img_path = base_dir_path + base_dir_path_ablation + dataset_mapping_dict[train_new_name] + "rgb"
        if train_dataset_name != test_dataset_name:
            register_coco_instances(train_dataset_name, {}, json_path, img_path)

    if test_new_name in dataset_mapping_dict.keys():
        json_path = base_dir_path + base_dir_path_ablation + dataset_mapping_dict[test_new_name] + json_add
        img_path = base_dir_path + base_dir_path_ablation + dataset_mapping_dict[test_new_name] + "rgb"
        register_coco_instances(test_dataset_name, {}, json_path, img_path)
    pass

def main(args):
    """ My steps """
    print('Start registering......')
    my_register_coco_dataset(args)
    print('Register done!')


    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
