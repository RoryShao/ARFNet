GPU_ID=6
# VISDA-C

python image_target.py --gpu_id $GPU_ID --net ARFNet101 --batch_size 64  --max_epoch 50 --lr 0.003 --dset  VISDA-C --da uda  --ent --gent --label_ways  dynamic_label --glcl --glcl_co 0.1 --distill

python image_target.py --gpu_id $GPU_ID --net vit_base --batch_size 64  --max_epoch 100 --lr 0.003 --dset  VISDA-C --da uda  --ent --gent --label_ways  dynamic_label --glcl --glcl_co 0.3 --distill


python image_target.py --gpu_id $GPU_ID --net avit_base --batch_size 64  --max_epoch 100 --lr 0.003 --dset  VISDA-C --da uda  --ent --gent --label_ways  dynamic_label --glcl --glcl_co 0.3 --distill
