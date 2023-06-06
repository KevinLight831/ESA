import logging
from lib import evaluation
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# save results
os.system("python3 eval.py --dataset f30k --data_path ../../data/f30k --model_name runs/f30k_butd_region_bert --save_results")
os.system("python3 eval.py --dataset f30k --data_path ../../data/f30k --model_name runs/f30k_butd_region_bert1 --save_results")

# Evaluate model ensemble
paths = ['runs/f30k_butd_region_bert/results_f30k.npy',
         'runs/f30k_butd_region_bert1/results_f30k.npy']
logger.info('------------------------------------ensemble-------------------------------------------------')
#evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)
