export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM="true"
python src/fashion_clip_finetune.py \
   --dataset fashioniq \
   --num-epochs 5 \
   --clip-model-name RN50x4 \
   --encoder both \
   --learning-rate 5e-6 \
   --batch-size 64 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1