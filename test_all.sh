#!/bin/bash

#SBATCH --job-name test_sr50
#SBATCH --partition dios
#SBATCH --gres=gpu:1

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/apalencia/code/envs/tfgpu1.12py3.6.13
export TFHUB_CACHE_DIR=.

start=`date +%s`

python test_all.py

end=`date +%s`

runtime=$((end-start))

mail -s "Proceso finalizado" e.alepalenc@go.ugr.es <<< "El proceso test_all (30000 samples) ha finalizado en $[ $runtime/3600 ] h. $[ ($runtime%3600)/60 ] min. $[ $runtime%60 ] seg."
