GPU_ID=4
# office31

# CNN 
for s in 0 1 2 
do
    echo "s=$s"
    python image_source.py --gpu_id $GPU_ID --net ARFNet50 --cl_co 0.1 --batch_size 64 --max_epoch 200 --lr 0.03 --da uda  --s $s --data_root data/ --distill
done
echo

# ViT
for s in 0 1 2 
do
    echo "s=$s"
    python image_source.py --gpu_id $GPU_ID --net vit_base --cl_co 0.1 --batch_size 64 --max_epoch 50 --lr 0.03 --da uda  --s $s --data_root data/ # --distill
done
echo



# arfnet
for s in 0 1 2 
do
    echo "s=$s"
    python image_source.py --gpu_id $GPU_ID --net avit_base --cl_co 0.1 --batch_size 64 --max_epoch 50 --lr 0.03 --da uda  --s $s --data_root data/ # --distill
done
echo
