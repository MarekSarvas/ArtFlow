python3 -u multi_style_transfer.py \
--gpu False \
--content_dir data/content --style_dir data/style --size 256 \
--n_flow 16 --n_block 1 --operator att --decoder models/att_16_1.pth \
--output interpolation_ArtFlow-AdaIN



