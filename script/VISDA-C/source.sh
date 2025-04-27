GPU_ID=6
# VISDA-C

python image_source.py --gpu_id $GPU_ID --net ARFNet101 --batch_size 64  --max_epoch 100 --lr 0.001 --da uda --dset VISDA-C --s 0 

python image_source.py --gpu_id $GPU_ID --net vit_base --batch_size 64  --max_epoch 200 --lr 0.001 --da uda --dset VISDA-C --s 0 

python image_source.py --gpu_id $GPU_ID --net avit_base --batch_size 64  --max_epoch 200 --lr 0.001 --da uda --dset VISDA-C --s 0 
