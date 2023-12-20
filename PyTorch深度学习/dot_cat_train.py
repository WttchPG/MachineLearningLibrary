import os.path
import time
from typing import Optional

import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


class DogCatModule:
    """
    Attributes:
        full_images (Optional[ImageFolder]): 数据集 ImageFolder
        train_data_loader (DataLoader): 训练数据集
        test_data_loader (DataLoader): 测试数据集
        resnet (nn.Module): resnet 预训练模型
        learning_rate (float): 学习率
        cross_entropy_loss_fn (nn.CrossEntropyLoss): 损失函数
        device (str): 训练设备
    """

    full_images: Optional[ImageFolder]
    train_data_loader: DataLoader
    test_data_loader: DataLoader
    resnet: nn.Module
    learning_rate: float
    cross_entropy_loss_fn: nn.CrossEntropyLoss
    device: str
    simple_transform: Compose

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'train on {self.device}')

        # 数据转换，图片转换为张量并正则化
        self.simple_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 损失函数和优化器
        self.learning_rate = 0.001
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()

        self._load_resnet()

        if torch.cuda.device_count() > 1:
            self.resnet = nn.DataParallel(self.resnet)
            self.resnet = self.resnet.to(self.device)

    @staticmethod
    def is_valid_file(file: str) -> bool:
        """
        校验文件是否满足要求
        Args:
            file (str): 文件名称
        Returns:
            bool: True 如果文件满足条件, 否则 False
        """
        return not file.split('/')[-1].startswith("._") and file.endswith('jpg')

    def load_image(self, path: str):
        """
        加载数据集, 并转换为张量格式
        """
        self.full_images = ImageFolder(
            path,
            transform=self.simple_transform,
            is_valid_file=self.is_valid_file
        )

    def split_dataset(self, train_size=20000, batch_size=64, num_workers=3, shuffle=True):
        """
        分割数据集
        Args:
            train_size: 训练集大小
            batch_size: 批处理大小
            num_workers: 读取数据用的 worker 数量
            shuffle: 加载时是否乱序数据集
        """
        train_dataset, test_dataset = random_split(
            self.full_images,
            [train_size, len(self.full_images) - train_size]
        )  # type: Subset, Subset

        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle': shuffle
        }
        if torch.cuda.is_available():
            kwargs['pin_memory'] = True
            kwargs['pin_memory_device'] = 'cuda'
        self.train_data_loader = DataLoader(train_dataset, **kwargs)
        self.test_data_loader = DataLoader(test_dataset, **kwargs)

    def _load_resnet(self):
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
        num_in_features = self.resnet.fc.in_features

        self.resnet.fc = torch.nn.Linear(num_in_features, 2)

        self.resnet = self.resnet.to(self.device)

    def train(self, epoch=25):
        for i in range(epoch):
            begin_at = time.time()
            print(f'epoch [{i + 1}/{epoch}] start...')
            self.train_model()
            self.test_model()

            print(f'epoch [{i + 1}/{epoch}] end, use: {time.time() - begin_at}')

        torch.save(self.resnet.state_dict(), './resnet.pkl')

    def train_model(self):
        optimizer = optim.SGD(self.resnet.parameters(), lr=self.learning_rate, momentum=0.9)
        size = len(self.train_data_loader.dataset)

        # 开始训练
        self.resnet.train()

        for batch, (X, y) in enumerate(self.train_data_loader):
            # self._train_message_sender.epoch_start(batch + 1, size)
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            # 计算
            pred = self.resnet(X)
            # 损失
            loss = self.cross_entropy_loss_fn(pred, y)

            # 反向传播
            loss.backward()
            # 收集梯度
            optimizer.step()
            # 梯度归零
            optimizer.zero_grad()

            loss, current = loss.item(), (batch + 1) * len(X)

            if batch % 100 == 0:
                # 100 批打印一次
                print(
                    f'[{current}/{size}]: loss: {loss}  f1 score: {f1_score(y.cpu().numpy(), pred.argmax(1).cpu().numpy())}')
            # self._train_message_sender.epoch_finish(batch + 1, size, 0, loss)

            if torch.backends.mps.is_available():
                # 只训练一个 batch
                print("Train on Mac, break...")
                break

    def test_model(self):
        self.resnet.eval()
        size = len(self.test_data_loader.dataset)

        num_batches = len(self.test_data_loader)

        test_loss, correct = 0, 0

        with torch.no_grad():
            # 测试关闭梯度
            for X, y in self.test_data_loader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                pred = self.resnet(X)
                test_loss += self.cross_entropy_loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                if torch.backends.mps.is_available():
                    print("Test on Mac, break...")

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 100 * correct, test_loss


if __name__ == '__main__':
    image_path = "/home/wangc/datasets/CatsVsDogs/archive/PetImages"
    dog_cat_module = DogCatModule()
    if not os.path.exists('./resnet.pkl'):

        dog_cat_module.load_image(image_path)
        dog_cat_module.split_dataset()

        dog_cat_module.train(epoch=20)
    else:
        module = resnet18()
        num_in_features = module.fc.in_features

        module.fc = torch.nn.Linear(num_in_features, 2)
        state_dict = torch.load('./resnet.pkl', map_location=torch.device('mps'))
        state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict}
        module.load_state_dict(state_dict)

        simple_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_tensor = Image.open('./2.jpeg')
        test_tensor = simple_transform(test_tensor)

        test_tensor1 = torch.zeros((64, 3, 224, 224))
        test_tensor1[1] = test_tensor

        print(module(test_tensor1))
