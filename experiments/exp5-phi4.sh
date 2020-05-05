echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
echo $LD_LIBRARY_PATH
srun -p ls6 --gres=gpu:1 python3 -u ../train.py --snapshot ../pretrained/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5 --phi 4 --epochs 200 --compute-val-loss --random-transform --tensorboard-dir ../logs/exp5_phi4 --snapshot-path ../checkpoints/exp5_phi4 csv-rand-val ../../../polyp-datasets/train_polyp_bb.csv ../../../polyp-datasets/class_id.csv --val-fraction=0.1