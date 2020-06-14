NUM_PROC=$1
shift
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"  ../../imagenet/ILSVRC/Data/CLS-LOC/ \
    --model vgg_bn_ori_quant_v1 --num-gpu 1 \
    -b 32 -j 16 --lr 0.01 --drop 0.2 --img-size 224 --sched step --epochs 120 \
    --decay-epochs 30 --decay-rate 0.1 --opt sgd --warmup-epochs 0 --weight-decay 1e-4 --opt-eps .001 --batch-size 32 --log-interval 50
