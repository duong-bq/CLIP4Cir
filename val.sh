python src/validate.py \
   --dataset fashioniq \
   --combining-function sum \
   --combiner-path checkpoints/fiq_comb_RN50x4_fullft.pt \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path checkpoints/fiq_clip_RN50x4_fullft.pt \
   --target-ratio 1.25 \
   --transform targetpad