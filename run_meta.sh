# Bash template for training and testing different models
# Author: Marek Sarvas
#
# =================== Train configs ====================

stage=3
stop_stage=3
verbose=0
ngpus=1
 
exp_id=$EXP_ID
                       
# =================== PATHS ====================
MAIN_PATH=${PWD}
TRAIN_DATA_PATH=${MAIN_PATH}/data/train
                                               
STYLE_PATH=${MAIN_PATH}/data/style
CONTENT_PATH=${MAIN_PATH}/data/content            
                                   
if [ -n "$BASE" ]; then
    EXP_FOLDER=${BASE}/experiment
else
    EXP_FOLDER=${MAIN_PATH}/experiment
fi


mkdir -pv ${TRAIN_DATA_PATH}
mkdir -pv ${EXP_FOLDER}

# load modules
module add anaconda3-4.0.0
module add conda-modules-py37

# activate (or create) the conda environment
conda activate torch_env || {
  conda create -n torch_env python=3.7;
  conda activate torch_env;
  conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
}

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
        --n_flow 8 --n_block 2 
        --operator wct 
        --decoder experiments/ArtFlow-AdaIN/glow.pth \
        --output output_ArtFlow-AdaIN
fi

if [ ${stage} -le 3 ] &&[ ${stop_stage} -ge 3 ]; then
   echo "stage 3: Train Glow with AdaAttN"
    python -u train.py \
        --content_dir "${TRAIN_DATA_PATH}/coco"  \
        --style_dir "${TRAIN_DATA_PATH}/wikiart" \
        --save_dir=${EXP_FOLDER}/${exp_id} \
        --n_flow $N_FLOW       \
        --n_block $N_BLOCK      \
        --batch_size $BATCH_SIZE   \
        --max_iter $MAX_ITER    \
        --operator att
fi
# TODO: update parameters
if [ ${stage} -le 4 ] &&[ ${stop_stage} -ge 4 ]; then
   echo "stage 4: Test Glow with AdaAttN"
    python -u test.py \
        --content_dir "${CONTENT_PATH}"  \
        --style_dir "${STYLE_PATH}" \
        --save_dir=${EXP_FOLDER}/${exp_id} \
        --size 256 \
        --n_flow 8 --n_block 2 
        --operator wct 
        --decoder experiments/ArtFlow-AdaIN/glow.pth \
        --output output_ArtFlow-AdaIN
fi
