SEED=1234   # random seed to produce deterministic results 
OBJ=edp     # optimization objective
NZ=4        # latent dimension size  
EPOCHS=2000 # training epoches  
DATASET_SIZE=131328 # specify training datasize for the target dataset 
PRED_MODEL=orig_1 # predictor model def 
VAE_MODEL=model_1 # vae model def
DATASET_PATH=../db/dataset_all_layer.csv 
OBJ=edp
DNN_DEF=../workloads/test_layers/layer_1.json
SEARCH_SAMPLES=1

SEARCH_SEEDS="1234"

NAME=example #dataset_${DATASET_SIZE}_seed_${SEED}_nz_${NZ}_model_${MODEL}_target_${TARGET_MODEL}_vae_${VAE_MODEL}
APPENDIX=_VAE_${NAME}
OUTPUT_DIR=results/cosa_data${APPENDIX}_predictor
mkdir -p $OUTPUT_DIR
cp ../model/model.pth $OUTPUT_DIR/model_checkpoint2500.pth
for SEARCH_SEED in ${SEARCH_SEEDS}; do
    echo ${APPENDIX}
    python train.py --data-name cosa_data --save-interval 100 --save-appendix ${APPENDIX}  --epochs ${EPOCHS} --lr 1e-4 --model VAE --predictor --nz ${NZ} --batch-size 16 --data-type cosa  --seed ${SEARCH_SEED} --dataset-size ${DATASET_SIZE} --predictor-model ${PRED_MODEL} --dataset-path ${DATASET_PATH} --continue-from 2500 --only-dnn-search --new-dnn-path ${DNN_DEF}  --search-samples ${SEARCH_SAMPLES} --obj ${OBJ} --VAE-model ${VAE_MODEL}  --search-lr 1e-1
done 
