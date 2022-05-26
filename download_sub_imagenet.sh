rm -rf data/eval/

git clone https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git
mkdir -p data/eval/
mv Small-ImageNet-Validation-Dataset-1000-Classes/ILSVRC2012_img_val_subset/  data/eval
mv Small-ImageNet-Validation-Dataset-1000-Classes/generator.py eval/
mv Small-ImageNet-Validation-Dataset-1000-Classes/imagenet_class_index.json eval/
mv Small-ImageNet-Validation-Dataset-1000-Classes/imagenet.json eval/
rm -rf Small-ImageNet-Validation-Dataset-1000-Classes
