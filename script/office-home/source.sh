GPU_ID=6
# office-home

for s in 0 1 2 3
do
    echo "s=$s"
python image_source.py --gpu_id $GPU_ID --net ARFNet50 --batch_size 64 --max_epoch 100 --lr 0.001 --da uda  --dset office-home --s $s --data_root data/ --distill
done
echo

for s in 0 1 2 3
do
    echo "s=$s"
python image_source.py --gpu_id $GPU_ID --net vit_base --batch_size 64 --max_epoch 200 --lr 0.003 --da uda  --dset office-home --s $s --data_root data/ 
done
echo



for s in 0 1 2 3
do
    echo "s=$s"
python image_source.py --gpu_id $GPU_ID --net avit_base --batch_size 64 --max_epoch 200 --lr 0.003 --da uda  --dset office-home --s $s --data_root data/ --distill
done
echo