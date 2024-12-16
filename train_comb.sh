export CUDA_VISIBLE_DEVICES=1
python src/combiner_train.py \
   --dataset 'FashionIQ' \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 30 \
   --clip-model-name RN50x4 \
   --clip-model-path models/clip_finetuned_on_fiq_RN50x4_2024-12-12_16-12-07/saved_models/tuned_clip_best.pt \
   --combiner-lr 2e-5 \
   --batch-size 512 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1