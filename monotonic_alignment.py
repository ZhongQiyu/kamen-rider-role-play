import torch
from monotonic_align import monotonic_align

# 示例输入
input_sequence = torch.randn(10, 256)  # 10个时间步，特征维度256
target_sequence = torch.randn(5, 256)   # 5个时间步，特征维度256

# 对齐操作
alignments = monotonic_align(input_sequence, target_sequence)

# 使用对齐结果
print(alignments)

# alignment