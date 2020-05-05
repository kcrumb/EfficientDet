echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
echo $LD_LIBRARY_PATH
srun -p ls6 --gres=gpu:1 python3 -u ../train.py --snapshot ../pretrained/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5 --phi 2 --epochs 200 --compute-val-loss --random-transform --tensorboard-dir ../logs/exp3_phi2 --snapshot-path ../checkpoints/exp3_phi2 csv-rand-val ../../../polyp-datasets/train_polyp_bb.csv ../../../polyp-datasets/class_id.csv --val-fraction=0.1