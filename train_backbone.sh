mkdir train_backbone_result3
python train_backbone.py --predict role --use-wandb --gpu 2 --output-dir train_backbone_result3

mkdir train_backbone_result4
python train_backbone.py --predict frame --use-wandb --gpu 2 --output-dir train_backbone_result4