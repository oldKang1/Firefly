---
title: 深度学习项目模板
published: 2026-07-13
pinned: false
description: 学习
tags: [深度学习]
category: 深度学习
author: lyk
draft: false
---

project/
├── data.py          # 1. 数据准备（Dataset, DataLoader）
├── model.py         # 2. 模型结构（神经网络定义）
├── train.py         # 3. 核心训练与验证逻辑
├── config.py        # 4. 超参数配置
└── predict.py       # 5. 推理与测试

1. config.py（参数配给中心）
别把学率、Batch Size 硬编码在代码里，统一放这里。

Python
import torch

class Config:
    # 硬件配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    
    # 路径配置
    DATA_DIR = "./data"
    MODEL_SAVE_PATH = "./best_model.pth"

2. data.py（数据流水线）
负责把你的原始数据变成 PyTorch 能直接迭代的 DataLoader。

Python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """在这里读取你自己的数据集（文本、图片、表格等）"""
    def __init__(self, data_dir, is_train=True):
        super().__init__()
        #  【填空】：在这里加载原始数据
        # self.features = ...
        # self.labels = ...
        pass

    def __len__(self):
        #  【填空】：返回数据集的总样本数
        return 1000 

    def __getitem__(self, idx):
        #  【填空】：根据索引返回单个样本的 (x, y)
        # x = self.features[idx]
        # y = self.labels[idx]
        # return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        return torch.randn(10), torch.randint(0, 2, (1,)).item()

def get_loaders(config):
    """生成训练和验证的 DataLoader"""
    train_dataset = CustomDataset(config.DATA_DIR, is_train=True)
    val_dataset = CustomDataset(config.DATA_DIR, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

3. model.py（网络架构）
只负责网络的前向传播逻辑，不要塞入任何训练代码。

Python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #  【填空】：定义网络的每一层组件
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 假设是个2分类任务
        )
        
    def forward(self, x):
        #  【填空】：编排前向传播的顺序
        return self.network(x)

4. train.py（核心训练引擎）
这是最标准的标准训练流：前向传播 -> 计算损失 -> 梯度清零 -> 反向传播 -> 优化器迭代。

Python
import torch
import torch.nn as nn
from config import Config
from data import get_loaders
from model import MyNetwork

def train_one_epoch(model, loader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
        
        # 核心 5 步法
        outputs = model(inputs)            # 1. 前向传播
        loss = criterion(outputs, targets) # 2. 计算损失
        
        optimizer.zero_grad()              # 3. 梯度清零
        loss.backward()                    # 4. 反向传播
        optimizer.step()                   # 5. 更新参数
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

@torch.no_grad() # 验证时关闭梯度计算，省显存
def validate(model, loader, criterion, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        # 顺便算个准确率（如果是分类任务的话）
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100.0 * correct / total
    return running_loss / len(loader), acc


def main():
    config = Config()
    
    # 1. 初始化数据、模型、损失函数和优化器
    train_loader, val_loader = get_loaders(config)
    model = MyNetwork().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # 2. 循环 Epoch 训练
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config)
        val_loss, val_acc = validate(model, val_loader, criterion, config)
        
        print(f"Epoch [{epoch}/{config.EPOCHS}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 3. 保存最佳模型权重（Early Stopping 的雏形）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f" Saved best model with Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()