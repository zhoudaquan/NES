CUDA_VISIBLE_DEVICES=0 python train.py ../../imagenet/ILSVRC/Data/CLS-LOC/ --model mobilenetv2_100 --num-gpu 1 -j 1 --lr 0.12 --drop 0.2 --img-size 224 --sched step --epochs 700 --decay-epochs 3 --decay-rate 0.97 --opt rmsproptf --warmup-epochs 5 --warmup-lr 1e-6 --weight-decay 1e-5 --opt-eps .001 --model-ema --batch-size 64 --log-interval 1 --fc_compress "fully_fc" --enable_se --up_sampling_ratio 1.0 --eval-only -tb 1 --no-args