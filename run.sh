# Bash template for training and testing different models
# Author: Marek Sarvas
#
# =================== Train configs ====================

stage=5
stop_stage=5
verbose=0
ngpus=1
 
exp_id=test2
                       
# =================== PATHS ====================
MAIN_PATH=${PWD}
TRAIN_DATA_PATH=${MAIN_PATH}/data/train
                                               
STYLE_FOLDER=comparison_data/style
CONTENT_FOLDER=comparison_data/content

STYLE_PATH=${MAIN_PATH}/${STYLE_FOLDER}
CONTENT_PATH=${MAIN_PATH}/${CONTENT_FOLDER}            
                                   
EXP_FOLDER=${MAIN_PATH}/exp

mkdir -pv ${TRAIN_DATA_PATH}
mkdir -pv ${EXP_FOLDER}

if [ ${stage} -le 1 ] &&[ ${stop_stage} -ge 1 ]; then
   echo "stage 1: Train baseline Glow with AdaIn"
    python -u train.py \
        --content_dir "${TRAIN_DATA_PATH}/coco"  \
        --style_dir "${TRAIN_DATA_PATH}/wikiart" \
        --save_dir=${EXP_FOLDER}/${exp_id} \
        --n_flow 8       \
        --n_block 2      \
        --batch_size 1   \
        --operator adain
fi

if [ ${stage} -le 2 ] &&[ ${stop_stage} -ge 2 ]; then
   echo "stage 2: Test baseline Glow with AdaIn"
    python -u test.py \
        --content_dir "${CONTENT_PATH}"  \
        --style_dir "${STYLE_PATH}" \
        --size 256 \
        --n_flow 8 --n_block 2 \
        --operator wct  \
        --decoder experiments/ArtFlow-AdaIN/glow.pth \
        --output output_ArtFlow-AdaIN
fi

if [ ${stage} -le 3 ] &&[ ${stop_stage} -ge 3 ]; then
   echo "stage 3: Train Glow with AdaAttN"
    python -u train.py \
        --content_dir "${TRAIN_DATA_PATH}/coco"  \
        --style_dir "${TRAIN_DATA_PATH}/wikiart" \
        --save_dir=${EXP_FOLDER}/${exp_id} \
        --n_flow 4       \
        --n_block 2      \
        --batch_size 1   \
        --operator att
fi

if [ ${stage} -le 4 ] &&[ ${stop_stage} -ge 4 ]; then
   echo "stage 4: Test Glow with AdaAttN"
   export CUDA_VISIBLE_DEVICES=0
   python3 -u test.py \
        --gpu False \
        --pad 0 \
        --content_dir "${CONTENT_PATH}"  \
        --style_dir "${STYLE_PATH}" \
        --size 256 \
        --n_flow 8 --n_block 2 --n_trans 1  --max_sample 96 \
        --operator att \
        --decoder models/artflow_8_1_2.pth \
        --output ${EXP_FOLDER}/adaatt_nopad_8_1_2_96
fi

if [ ${stage} -le 5 ] &&[ ${stop_stage} -ge 5 ]; then
   echo "stage 4: Test Glow with AdaAttN"
   export CUDA_VISIBLE_DEVICES=0
   python3 -u comparison.py \
        --gpu False \
        --pad 64 \
        --content_dir "${CONTENT_PATH}"  \
        --style_dir "${STYLE_PATH}" \
        --size 256 \
        --n_flow 8 --n_block 2 --n_trans 1  --max_sample 96 \
        --operator att \
        --decoder models/artflow_8_1_2.pth \
        --output ${EXP_FOLDER}/comparison
fi

