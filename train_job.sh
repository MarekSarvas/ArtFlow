#!/bin/bash    
#PBS -N artflow_train     
#PBS -l select=1:ncpus=2:mem=32gb:scratch_local=150gb:ngpus=1:cl_adan=True    
#PBS -q gpu    
#PBS -l walltime=15:00:00    
#PBS -m ae


# define BASE TODO: alter..
export BASE=/storage/brno2/home/pirxus/
DATA_TRAIN=$BASE/artflow_data/train
DATA=$SCRATCHDIR/data     


# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually     
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt    
     
# test if scratch directory is set    
# if scratch directory is not set, issue error message and exit    
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }    


# load modules    
module add anaconda3-4.0.0    
module add conda-modules-py37    
    
# activate (or create) the conda environment    
conda activate torch_env || {    
  conda create -n torch_env python=3.7;    
  conda activate torch_env;    
  conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch    
  pip install kaldiio;    
}

mkdir $SCRATCHDIR/data     
    
# copy the train and test data    
echo "Copying training data.."
cp -r $DATA_TRAIN $SCRATCHDIR/data/ || {    
  echo >&2 "Error while copying training data!";    
  exit 2;    
}

# move into scratch directory
cd $SCRATCHDIR

# copy the repository..
cp -r $BASE/ArtFlow .
mv $SCRATCHDIR/data/train ArtFlow/data/
cd ArtFlow

# run the training..
bash run_meta.sh

# clean the SCRATCH directory and exit :)
clean_scratch
