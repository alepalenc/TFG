#!/bin/bash

#SBATCH --job-name train_sr50
#SBATCH --partition dios
#SBATCH --gres=gpu:1

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/apalencia/code/envs/tfgpu1.12py3.6.13
export TFHUB_CACHE_DIR=.

start=`date +%s`

python neurosat_gpu.py \
--use_tpu=False \
--train_file=sr_50/train_1_sr_50.tfrecord \
--test_file=sr_50/train_2_sr_50.tfrecord \
--train_steps=600000 \
--test_steps=80 \
--model_dir=model_sr50 \
--export_dir=export_sr50 \
--variable_number=50 \
--clause_number=500 \
--train_files_gzipped=False \
--batch_size=64 \
--export_model \
--attention=False

end=`date +%s`

runtime=$((end-start))

mail -s "Proceso finalizado" e.alepalenc@go.ugr.es <<< "El proceso train_sr50 ha finalizado en $[ $runtime/3600 ] h. $[ ($runtime%3600)/60 ] min. $[ $runtime%60 ] seg."
