# Experiment Note Record for Dynamic Weight Change -- step 1.2 a

每个训练步骤：

1. 所有模型开始时权重 = 1.0 (平等)
2. 执行rollout → 获得各模型表现
3. 立即计算新权重 → 应用到policy update
4. 下一步重置权重为1.0，重新开始

## How to Calculate the Weight

**attempt1**: 排名法

1. add a new file train_weight.py 