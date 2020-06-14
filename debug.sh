NUM_PROC=$1
shift
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_grad_cam.py "$@"  \
    ../../imagenet/ILSVRC/Data/CLS-LOC/ \
    --model i2rnetv3 --num-gpu 1 -j 16 \
    --lr 0.1 --drop 0.2 --img-size 224 --sched cosine \
    --epochs 200 --decay-epochs 30 --decay-rate 0.1 --opt sgd \
    --warmup-epochs 0 --weight-decay 1e-4 --opt-eps .001 --batch-size 1 --log-interval 500 \
    --eval-only \
    --resume timm/models/74_02_full_id_tensor/model_best.pth.tar
