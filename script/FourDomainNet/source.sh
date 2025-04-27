GPU_ID=7
#  ['clipart', 'painting', 'real', 'sketch']
for s in 0 1 2 3
do
    echo "s=$s"
    python source_fourdomainnet.py --gpu_id $GPU_ID --epoches 50  --s $s --net ARFNet50 
done
echo


for s in 0 1 2 3
do
    echo "s=$s"
    python source_fourdomainnet.py --gpu_id $GPU_ID --epoches 50  --s $s --net vit_base 
done
echo

for s in 0 1 2 3
do
    echo "s=$s"
    python source_fourdomainnet.py --gpu_id $GPU_ID --epoches 50  --s $s --net avit_base 
done
echo