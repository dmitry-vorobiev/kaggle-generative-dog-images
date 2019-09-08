#!/bin/bash
python sample.py \
--data_root ./input/all-dogs \
--label_root ./input/annotation/Annotation \
--num_epochs 60 --shuffle --num_workers 2 --batch_size 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_ch 32 --D_ch 64 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--G_shared \
--hier --dim_z 100 --shared_dim 128 \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--G_init ortho --D_init ortho \
--G_eval_mode \
--ema --use_ema --ema_start 30000 \
--save_every 5000 --num_save_copies 0 --seed 1234 \
--sample_num 10000 --trunc_z 0.0