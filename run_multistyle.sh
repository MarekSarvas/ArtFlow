python3 -u multi_style_transfer.py \
--gpu False \
--content_dir interpolation_data/content --style_dir interpolation_data/style --size 256 \
--n_flow 8 --n_block 2 --operator att --decoder models/att_8_2.pth \
--output interpolation_ArtFlow-AdaIN



