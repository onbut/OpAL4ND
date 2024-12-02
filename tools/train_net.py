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

def register_from_path(dataset_name):
    # base_dir_path = "/home/yagao/workspace/openset/mydete/data_double/"
    base_dir_path = "../data_double/"

    json_add = "json/coco_1.json"
    if "U4" in dataset_name:
        json_add = "json/coco_4.json"
        base_dir_path = base_dir_path + "U4/"
    elif "U3" in dataset_name:
        json_add = "json/coco_5.json"
        base_dir_path = base_dir_path + "U3/"
    elif "U2" in dataset_name:
        json_add = "json/coco_6.json"
        base_dir_path = base_dir_path + "U2/"
        pass

    if "01" in dataset_name:
        json_add = "json/coco_01.json"
        pass

    if "no_p" in dataset_name:
        base_dir_path = base_dir_path + "data_no_p/"
    elif "ucl" in dataset_name:
        base_dir_path = base_dir_path + "data_ucl/"
    elif "oic" in dataset_name:
        base_dir_path = base_dir_path + "data_oic/"
    elif "upl" in dataset_name:
        base_dir_path = base_dir_path + "data_upl/"
    elif "no_ic" in dataset_name:
        base_dir_path = base_dir_path + "data_no_ic/"
    elif "no_il" in dataset_name:
        base_dir_path = base_dir_path + "data_no_il/"
    elif "opu" in dataset_name:
        base_dir_path = base_dir_path + "data_opu/"
    elif "ran" in dataset_name:
        base_dir_path = base_dir_path + "data_ran/"
    elif "cen" in dataset_name:
        base_dir_path = base_dir_path + "data_cen/"
    elif "unc" in dataset_name:
        base_dir_path = base_dir_path + "data_unc/"
    elif "cor" in dataset_name:
        base_dir_path = base_dir_path + "data_cor/"
    else:
        base_dir_path = base_dir_path + "data_all_three/"
        pass

    mid_path = "coco_base_all_418/"
    if "bases_80" in dataset_name:
        mid_path = "coco_base_test_80/"
    elif "bases_338" in dataset_name:
        mid_path = "coco_base_train_338/"
    elif "_18" in dataset_name:
        mid_path = "coco_labeled_18/"
    elif "_68" in dataset_name:
        mid_path = "coco_labeled_68/"
    elif "_118" in dataset_name:
        mid_path = "coco_labeled_118/"
    elif "_168" in dataset_name:
        mid_path = "coco_labeled_168/"
    elif "_218" in dataset_name:
        mid_path = "coco_labeled_218/"
    elif "_268" in dataset_name:
        mid_path = "coco_labeled_268/"
    elif "_318" in dataset_name:
        mid_path = "coco_labeled_318/"
    elif "_20" in dataset_name:
        mid_path = "coco_unlabeled_20/"
    elif "_70" in dataset_name:
        mid_path = "coco_unlabeled_70/"
    elif "_120" in dataset_name:
        mid_path = "coco_unlabeled_120/"
    elif "_170" in dataset_name:
        mid_path = "coco_unlabeled_170/"
    elif "_220" in dataset_name:
        mid_path = "coco_unlabeled_220/"
    elif "_270" in dataset_name:
        mid_path = "coco_unlabeled_270/"
    elif "_320" in dataset_name:
        mid_path = "coco_unlabeled_320/"
    
    
    img_path = base_dir_path + mid_path + "rgb"
    json_path = base_dir_path + mid_path + json_add
    print(f"Registering {dataset_name} from {json_path}")
    register_coco_instances(dataset_name, {}, json_path, img_path)
    pass


# My functions
def my_register_coco_dataset(args):
    cfg_data = get_cfg()
    cfg_data.merge_from_file(args.config_file)

    # 获取数据集的名称
    train_dataset_name = cfg_data.DATASETS.TRAIN[0]
    test_dataset_name = cfg_data.DATASETS.TEST[0]

    # 注册数据集
    register_from_path(test_dataset_name)
    if train_dataset_name != test_dataset_name:
        register_from_path(train_dataset_name)
    
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
