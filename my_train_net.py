"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for other models.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # 1. Загрузка базового конфига, указанного в аргументах (например, mask_rcnn_R_50_FPN_3x.yaml)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # 2. Настройка пользовательского датасета
    # Регистрируем датасеты, если они еще не зарегистрированы
    # Указываем абсолютные или относительные пути к json и папкам с изображениями
    register_coco_instances("my_dataset_train", {}, "data/coco/annotations/instances_train.json", "data/coco/train")
    register_coco_instances("my_dataset_val", {}, "data/coco/annotations/instances_val.json", "data/coco/val")
    
    # Устанавливаем метаданные вручную, чтобы точно совпадало с ID в JSON
    MetadataCatalog.get("my_dataset_train").thing_classes = ["frame"]
    MetadataCatalog.get("my_dataset_val").thing_classes = ["frame"]
    
    # 3. Переопределение параметров для дообучения
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    # Количество рабочих процессов загрузчика данных (можно уменьшить для Windows)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Важно для RLE масок
    cfg.INPUT.MASK_FORMAT = "bitmask" 
    
    # Настройки модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Только класс "frame"
    
    # Настройки солвера (примерные, можно переопределить через args)
    # Предполагаем обучение на 1 GPU, поэтому корректируем LR и Batch Size
    # Стандартные конфиги рассчитаны на 8 GPU и batch size 16 (2 на GPU)
    # Здесь ставим 2 изображения на 1 GPU
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # Уменьшаем LR пропорционально (стандарт 0.02 для bs=16)
    
    # Путь для сохранения весов
    cfg.OUTPUT_DIR = "./output"
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may be overwritten to obtain custom behaviors.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
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

# ---
# Пример запуска:
# python my_train_net.py --num-gpus 1 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
