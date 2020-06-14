NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"  ../../imagenet/ILSVRC/Data/CLS-LOC/ --model darts_searched --num-gpu 1 -j 16 --lr 0.1 --drop 0.2 --img-size 224 --sched step --epochs 650 --decay-epochs 1 --decay-rate 0.97 --opt sgd --warmup-epochs 5 --warmup-lr 1e-6 --weight-decay 1e-5 --opt-eps .001 --batch-size 64 --log-interval 500 --fc_compress "fully_fc" --enable_se --group_se --up_sampling_ratio 1.0  --model-ema --display-info
