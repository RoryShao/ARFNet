GPU_ID=5
# office-home

for s in  0 1 2 3
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net ARFNet50 --batch_size 64  --max_epoch 200 --lr 0.001 --dset  office-home --da uda --s $s --ent --gent --issave False \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.3 \
    --distill
done
echo


for s in  0 1 2 3
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net vit_base --batch_size 64  --max_epoch 200 --lr 0.001 --dset  office-home --da uda --s $s --ent --gent --issave False \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.3 \
    --distill
done
echo


for s in  0 1 2 3
do
    echo "$s"
    python image_target.py --gpu_id $GPU_ID --net avit_base --batch_size 64  --max_epoch 200 --lr 0.001 --dset  office-home --da uda --s $s --ent --gent --issave False \
    --label_ways  dynamic_label \
    --glcl --glcl_co 0.3 \
    --distill
done
echo
