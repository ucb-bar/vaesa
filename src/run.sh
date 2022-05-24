SEED=1234
NZ=4
EPOCHS=2000
DATASET_SIZE=131328
PRED_MODEL=orig_1
VAE_MODEL=model_1
DATASET_PATH=../db/dataset_all_layer.csv 
OBJ=edp
python train.py --data-name cosa_data --save-interval 10  --epochs ${EPOCHS} --lr 1e-4 --model VAE --predictor --bidirectional --nz ${NZ} --batch-size 64 --data-type cosa  --train-from-scratch --seed ${SEED} --dataset-size ${DATASET_SIZE} --predictor-model ${PRED_MODEL} --dataset-path ${DATASET_PATH} --obj ${OBJ} --VAE-model ${VAE_MODEL} --reprocess
