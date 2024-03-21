#!/bin/bash
#SBATCH --account=nn10039k            
#SBATCH --job-name=DeepFormer_5e5
#SBATCH --nodes=1    
#SBATCH --mem=64G
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --ntasks=8
#SBATCH --time=10-00:00:00  # 10 days               
#SBATCH --mail-user=nguyen.thanh.dat@nmbu.no
#SBATCH --mail-type=ALL



## function

run_training_standard() {
    model=$1
    n_pad_windows=$2
    lr=$3
    spec=$4
    loss=$5
    echo "traning $spec $model, $lr, $loss"

    ## fixed_variables
    val_data='15_seq.val'
    outdir='train_results'

    img=/cluster/projects/nn10039k/dat/singularity/ndatth-pytorch-v0.0.0.img
    data_dir=/cluster/projects/nn10039k/dat/run_deeplearning/data_${spec}

    ## get training data array
    my_array=$(ls $data_dir/*seq.bin)

    ## make sue outdir exist
    mkdir -p $outdir

    append_string="/data/"
    train_data=""
    for s in $my_array; do
        s=$(basename $s)
        train_data+=$(echo " $append_string$s")
    done

    singularity exec --nv --bind $PWD:/work_dir --bind $data_dir:/data \
        $img python /work_dir/my_trainer.py \
        --train ${train_data} \
        --val /data/${val_data} \
        --model models.${model} \
        --out /work_dir/${outdir}/${spec}_${model}_${lr}_${loss} \
        --threads 8 \
        --lr ${lr} \
        --decay 1e-6 \
        --n_pad_windows ${n_pad_windows} \
        --batch_size 512 \
        --loss ${loss}

}


## running
## model
model='DeepFormer'
n_pad_windows='2'
lr='5e-5'
spec='salmon'

run_training_standard $model $n_pad_windows $lr $spec 'logit'

run_training_standard $model $n_pad_windows $lr $spec 'focal'

