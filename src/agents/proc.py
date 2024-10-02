# proc.py

import pandas as pd
import numpy as np

def modify_labels_and_sample_data(file_path):
    # 加载数据
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    # 随机选取30%的样本
    sampled_df = df.sample(frac=0.3, random_state=42)  # 使用固定的随机种子以保证可重复性
    
    # 更改标签：如果是1改为0，如果是0改为1
    sampled_df['label'] = sampled_df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    # 保存修改后的数据到新文件中
    sampled_df.to_csv('data/test.txt', index=False, sep='\t', encoding='utf-8')
    return sampled_df

# 假设文件路径
file_path = 'data/train.txt'

# 调用函数
modified_samples = modify_labels_and_sample_data(file_path)
print(modified_samples)
