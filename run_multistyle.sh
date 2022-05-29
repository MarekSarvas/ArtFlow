python3 -u multi_style_transfer.py \
--gpu False \
--pad 32 \
--content_dir docu_data/content --style_dir docu_data/style --size 256 \
--n_flow 8 --n_block 2 --operator att --decoder models/att_8_2.pth \
--output exp/interpolation_pad_ArtFlow-AdaAttN



