
#  ['clipart', 'painting', 'real', 'sketch']

GPU_ID=7


for s in 0 1 2 3
do
    echo "s=$s"
    python target_fourdomainnet.py --gpu_id $GPU_ID --net ARFNet50 --lr 0.001 --epoches 50 --s $s  --ent --gent --label_ways  dynamic_label --distill --glcl 
done
echo



for s in 0 1 2 3
do
    echo "s=$s"
    python target_fourdomainnet.py --gpu_id $GPU_ID --net vit_base --lr 0.001 --epoches 50 --s $s  --ent --gent --label_ways  dynamic_label --glcl  --distill 
done
echo


for s in 0 1 2 3
do
    echo "s=$s"
    python target_fourdomainnet.py --gpu_id $GPU_ID --net avit_base --lr 0.001 --epoches 50 --s $s  --ent --gent --label_ways  dynamic_label --glcl  --distill 
done
echo