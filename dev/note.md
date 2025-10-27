# QUESTION
1. 如果是256的话 很多都没有生成完成就结束了 那不行 得到512

2. 是否需要formatreward？ 因为box很多时候没有输出出来


目前只对threshold做了修改 在里面加上了更多饿print 和 gradient_checkpoint的功能


sudo docker run -it \
  --name grpo_s1 \
  --gpus all \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  -e PYTORCH_NO_CUDA_IPC=1 \
  -e C10_DISABLE_PIDFD_GETFD=1 \
  -v /home/local/PARTNERS/yz646/grpo_s1:/workspace \
  -w /workspace \
  --shm-size=16g \
  hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 \
  /bin/bash