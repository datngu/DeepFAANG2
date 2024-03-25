#!/bin/bash
#SBATCH --account=nn10039k            
#SBATCH --job-name=catt_eval 
#SBATCH --nodes=1    
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --ntasks=8
#SBATCH --time=3-00:00:00  # 1 days               
#SBATCH --mail-user=nguyen.thanh.dat@nmbu.no
#SBATCH --mail-type=ALL




run_validation() {
    model=$1
    n_pad_windows=$2
    lr=$3
    spec=$4
    loss=$5
    echo "running evaluation $spec $model, $lr, $loss"


    img=/cluster/projects/nn10039k/dat/singularity/ndatth-pytorch-v0.0.0.img
    data_dir=/cluster/projects/nn10039k/dat/run_deeplearning/data_${spec}

    singularity exec --nv --bind $PWD:/work_dir --bind $data_dir:/data \
        $img python /work_dir/my_evaluator_rc.py \
        --test /data/16_seq.test \
        --n_center_windows 1 \
        --n_pad_windows $n_pad_windows \
        --batch_size 128 \
        --threads 8 \
        --model models.${model} \
        --model_weight /work_dir/train_results/${spec}_${model}_${lr}_${loss}/best_model.th  \
        --out /work_dir/train_evaluations_rc/${spec}_${model}_${lr}_${loss}.pkl

}


# here only run cattle
spec='cattle'
all_learning_rates=("1e-3" "1e-4" "1e-5" "5e-4" "5e-4" "5e-6")

### 1k models
all_models=("DeepSEA" "DanQ" "DeepATT" "DeepFormer")
n_pad_windows='2'

for lr in "${all_learning_rates[@]}"; do
    for model in "${all_models[@]}"; do
        run_validation $model $n_pad_windows $lr $spec 'logit'
    done
done



