import logging
from lib import evaluation
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# save results
os.system("python3 eval.py --dataset f30k --data_path ../../data/f30k --model_name runs/f30k_best1 --save_results")
os.system("python3 eval.py --dataset f30k --data_path ../../data/f30k --model_name runs/f30k_best2 --save_results")
# Evaluate model ensemble
paths = ['runs/f30k_best1/results_f30k.npy',
         'runs/f30k_best2/results_f30k.npy']
print('-------------------------------------------------------------------------------------')
#evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)

print('---------------------------------coco----------------------------------------------------')
os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best1 --save_results")
os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best2  --save_results")
# Evaluate model ensemble
paths = ['runs/coco_best1/results_coco.npy',
         'runs/coco_best2/results_coco.npy']
print('-------------------------------------------------------------------------------------')
evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)