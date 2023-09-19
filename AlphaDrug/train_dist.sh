#两卡训练示例脚本
source /opt/Ascend/ascend-toolkit/set_env.sh
cur_path=`pwd`
if [ $(uname -m) = "aarch64" ]
then
    #配置多卡端口
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500
    export WORLD_SIZE=4
    #配置多进程绑核
    for i in $(seq 0 3)
    do
	export LOCAL_RANK=$i
	let p_start=0+24*i
	let p_end=23+24*i
	#启动训练，参数根据训练代码进行自定义
	nohup taskset -c $p_start-$p_end python3 -u train_dist.py --layers 4 --bs 32 --local_rank=$i > ${cur_path}/train.log 2>&1 &
    done
else
    python3 -m torch.distributed.launch --nproc_per_node=2 train.py > ${cur_path}/train_x86.log 2>&1 &
fi

