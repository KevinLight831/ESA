import logging
from lib import evaluation
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# save results
os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_butd_region_bert --save_results")
os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_butd_region_bert1 --save_results")
# Evaluate model ensemble
paths = ['runs/coco_butd_region_bert/results_coco.npy',
         'runs/coco_butd_region_bert1/results_coco.npy']
logger.info('------------------------------------ensemble-------------------------------------------------')
evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)
