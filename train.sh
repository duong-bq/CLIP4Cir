export CUDA_VISIBLE_DEVICES=1
python src/clip_fine_tune.py \
   --dataset fashioniq \
   --num-epochs 15 \
   --clip-model-name RN50x4 \
   --encoder text \
   --learning-rate 5e-6 \
   --batch-size 512 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --plus