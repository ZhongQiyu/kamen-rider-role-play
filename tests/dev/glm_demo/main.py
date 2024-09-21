
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 导入我们自定义的 CUDA 扩展
import cuda_add

class MyDataset(Dataset):
    def __init__(self):
        self.x = np.random.rand(1000, 3)
        self.y = np.random.randint(low=0, high=2, size=(1000,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def collate_fn(batch):
    data_list, label_list = zip(*batch)
    return torch.tensor(data_list, dtype=torch.float32), torch.tensor(label_list, dtype=torch.long)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        # 使用我们自定义的 CUDA 加法操作
        weight = torch.ones_like(x).cuda()  # 创建一个与输入大小相同的 tensor
        out = cuda_add.add_forward(x, weight)[0]
        return self.fc(out)


if __name__ == "__main__":
    # 创建数据集和数据加载器
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

    # 初始化模型、损失函数和优化器
    model = SimpleModel().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        for data, label in dataloader:
            data = data.cuda()
            label = label.float().unsqueeze(1).cuda()  # 调整标签的尺寸

            # 前向传播
            outputs = model(data)

            # 计算损失
            loss = criterion(outputs, label)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item()}')

        # 使用 CUDA 同步
        torch.cuda.synchronize()
