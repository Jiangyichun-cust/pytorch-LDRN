import argparse
import torch
from torch import optim
from dataloader.dataloader import get_loader
from models.LDRN import Generator
from trainer import Trainer
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, weight):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)   # Mini-Batch大小，根据显存容量设置
    parser.add_argument('--lr', type=float, default=1e-4)   # 学习率
    parser.add_argument('--epochs', type=int, default=50)   # 总训练轮数
    parser.add_argument('--image-size', type=int, default=128)  # 裁剪后的高分辨率图像尺寸
    parser.add_argument('--scale', type=int, default=4)    # 放大倍数4倍
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = Generator().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    # 读取数据集，没有数据集可执行./datasets/prepare_datasets.py自动下载并crop
    dataloader = get_loader(opt.image_size, opt.scale, opt.batch_size, 1)

    trainer = Trainer(net_g, optimizer_g, dataloader, device, opt.batch_size, 10)

    writer = SummaryWriter('./log/' + 'pth')

    # 开始训练，监控训练过程于终端输入tensorboard --logdir "log" --samples_per_plugin=images=100
    for epoch in range(opt.epochs):
         if (epoch + 1) % 10 == 0:
            adjust_learning_rate(trainer.optimizer_g, 0.5)
         trainer.train(epoch, writer)

    writer.close()


if __name__ == '__main__':
    main()