
python validate.py /ssd/zhoudaquan/imagenet/ILSVRC/Data/CLS-LOC/ \
--model efficientnet_b0_dq \
--lr 0.12 -b 64 \
--drop 0.2 \
--img-size 224 \
--sched step \
--epochs 650 \
--decay-epochs 2 \
--decay-rate 0.97 \
--opt rmsproptf \
-j 32 \
--warmup-epochs 5 \
--warmup-lr 1e-6 \
--weight-decay 1e-5 \
--opt-eps .001 \
--num-gpu 1 \
--sampling \
--enable_se \
--resume ./output/train/75_5_model/model_75_5.pth.tar \
--log-interval 500