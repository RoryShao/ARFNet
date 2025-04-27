GPU_ID=4
# office31

for s in 0 1 2
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net ARFNet50 --batch_size 64  --max_epoch 200 --lr 0.003 --dset  office31 --da uda --s $s --ent --gent \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.1 \
    --distill
done
echo



for s in 0 1 2
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net vit_base --batch_size 64  --max_epoch 200 --lr 0.003 --dset  office31 --da uda --s $s --ent --gent \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.1 \
    --distill
done
echo

for s in 0 1 2
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net avit_base --batch_size 64  --max_epoch 200 --lr 0.003 --dset  office31 --da uda --s $s --ent --gent \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.1 \
    --distill
done
echo