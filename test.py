import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPU推荐使用NCCL
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 设置当前GPU


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


class ToyModel(torch.nn.Module):
    """简单的示例模型"""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def prepare_dataloader(rank, world_size, batch_size=32):
    """准备分布式数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    return dataloader


def train(rank, world_size):
    """每个GPU上运行的训练函数"""
    setup(rank, world_size)

    # 1. 准备模型
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 2. 准备数据
    train_loader = prepare_dataloader(rank, world_size)

    # 3. 准备优化器和损失函数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # 4. 训练循环
    for epoch in range(5):  # 训练5个epoch
        # 重要：设置epoch给sampler保证shuffle正常工作
        train_loader.sampler.set_epoch(epoch)

        ddp_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            data = data.view(data.size(0), -1)  # 展平MNIST图像

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:  # 只在主进程打印
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')

    # 5. 只在主进程保存模型
    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')

    cleanup()


def run_demo(demo_fn):
    """自动检测GPU并启动分布式训练"""
    world_size = torch.cuda.device_count()
    print(f"发现 {world_size} 个GPU，准备分布式训练...")

    mp.spawn(
        demo_fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # 自动检测所有GPU并启动训练
    run_demo(train)

    # 训练完成后，在主进程加载模型
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = ToyModel()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    print("模型加载完成！")
