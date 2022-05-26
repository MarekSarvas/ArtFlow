python3 -u multi_style_transfer.py \
--gpu False \
--content_dir data/content --style_dir data/style --size 256 \
--n_flow 8 --n_block 2 --operator wct --decoder experiments/ArtFlow-AdaIN/glow.pth \
--output interpolation_ArtFlow-AdaIN



