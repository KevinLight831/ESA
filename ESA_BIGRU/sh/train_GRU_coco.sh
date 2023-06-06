DATASET_NAME='coco'
DATA_PATH='../../data/'${DATASET_NAME}
VOCAB_PATH='../../data/vocab'
MODEL_NAME='runs/'${DATASET_NAME}'_butd_ESAregion_bigru'
cd ../
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name ${MODEL_NAME}/log --model_name ${MODEL_NAME} \
  --num_epochs=25 --lr_update=15 --learning_rate=.0005  --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 128 --hardnum 2

python3 eval.py --dataset ${DATASET_NAME}  --data_path ${DATA_PATH} --model_name ${MODEL_NAME}
