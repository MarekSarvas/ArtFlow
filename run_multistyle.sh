python3 -u multi_style_transfer.py \
--gpu False \
--pad 64 \
--content_dir docu_data/content --style_dir docu_data/style --size 256 \
--n_flow 16 --n_block 1 --n_trans 2 --operator att --decoder models/artflow_16_1_1.pth \
--output exp/interpolation_pad_ArtFlow-AdaAttN_16_1_1



