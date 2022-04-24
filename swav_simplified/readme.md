SINGLE NODE SINGLE GPU

To run:

`python -u main_swav.py --epochs 100 --batch_size 128 --base_lr 0.01 --final_lr 0.00001 --warmup_epochs 10 --start_warmup 0.3 --arch resnet18 --freeze_prototypes_niters 5000 --workers 2 --nmb_prototypes 1000`