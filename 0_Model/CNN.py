import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle


# 定义改进后的 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积 + 激活 + 批量归一化 + 池化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批量归一化
        self.pool = nn.MaxPool2d(2, 2)

        # 第二层卷积 + 激活 + 批量归一化 + 池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 批量归一化

        # 第三层卷积 + 激活 + 批量归一化 + 池化
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 批量归一化

        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 注意修改全连接层输入尺寸
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 卷积层1 + BatchNorm + ReLU + MaxPooling
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 卷积层2 + BatchNorm + ReLU + MaxPooling
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 卷积层3 + BatchNorm + ReLU + MaxPooling
        x = x.view(-1, 128 * 8 * 8)  # 展平操作，注意这里的尺寸要根据实际情况调整
        x = torch.relu(self.fc1(x))  # 全连接层1
        x = self.dropout(x)  # Dropout 层
        x = torch.relu(self.fc2(x))  # 全连接层2
        x = self.fc3(x)  # 输出层
        return x


# 数据加载与预处理
def prepare_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小
        transforms.ToTensor(),       # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])

    # 加载训练集和测试集
    train_data = datasets.ImageFolder(root=os.path.join(data_dir, '4 argumentation_train1'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(data_dir, '2 classified_test'), transform=transform)

    # 数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_data.classes


# 定义评估函数
def evaluate(model, data_loader, criterion, device):
    model.eval()  # 切换为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 新增：保存预测概率

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())  # 记录预测值
            all_labels.extend(labels.cpu().numpy())  # 记录真实标签

            # 记录预测概率
            probs = torch.softmax(outputs, dim=1)  # 使用 softmax 获取每一类的概率
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(data_loader)  # 平均损失
    accuracy = 100 * correct / total  # 准确率
    return avg_loss, accuracy, all_preds, all_labels, all_probs


# 定义训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, save_path, class_names, scheduler=None):
    best_accuracy = 0.0
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()  # 切换为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 记录训练集损失和准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 在测试集上评估模型
        test_loss, test_accuracy, all_preds, all_labels, _ = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # 打印当前 epoch 的训练和测试结果
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # 保存测试集效果最佳的模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"\n新最佳模型已保存，测试准确率: {best_accuracy:.2f}%")

            # 生成分类报告并保存

            # 使用 zero_division=1 来处理没有预测样本的类别
            report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=1)
            with open('best_classification_report.txt', 'w') as f:
                f.write(report)
            print("分类报告已保存。\n")

            # 生成并保存混淆矩阵图像
            confusion_matrix_path = 'confusion_matrix.png'
            plot_confusion_matrix(all_labels, all_preds, class_names, confusion_matrix_path)

        # 使用 ReduceLROnPlateau 调度器（如果有调度器）
        if scheduler:
            scheduler.step(test_loss)

    return train_losses, train_accuracies, test_losses, test_accuracies


# 绘制训练和测试的损失及准确率曲线
def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, save_path):
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, marker='o', label='Testing Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', label='Testing Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 保存图像
    print(f"训练和测试过程折线图已保存至 {save_path}")
    plt.show()


# 添加绘制混淆矩阵的函数
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 保存混淆矩阵图像
    print(f"混淆矩阵图像已保存至 {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_probs, class_names, save_path):
    """
    绘制 ROC 曲线
    :param y_true: 真实标签列表
    :param y_probs: 每一类的预测概率
    :param class_names: 类别名称列表
    :param save_path: ROC 图保存路径
    """
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))  # 将标签二值化

    fpr = dict()  # 存储每一类的 FPR
    tpr = dict()  # 存储每一类的 TPR
    roc_auc = dict()  # 存储每一类的 AUC

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], [p[i] for p in y_probs])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类别的 ROC 曲线
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # 绘制微平均 ROC 曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), [p for probs in y_probs for p in probs])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 保存图像
    print(f"ROC 曲线已保存至 {save_path}")
    plt.close()


# 主函数
if __name__ == "__main__":
    # 数据路径
    data_dir = '../'
    save_path = 'best_CNN_model.pth'

    # 训练模型
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # 准备数据
    train_loader, test_loader, class_names = prepare_data(data_dir, batch_size)

    # 初始化设备、模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7, verbose=True)

    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device, save_path, class_names, scheduler
    )

    # 在测试集上评估模型，生成 ROC 曲线
    _, _, _, all_labels, all_probs = evaluate(model, test_loader, criterion, device)
    plot_roc_curve(all_labels, all_probs, class_names, 'roc_curve.png')

    # 绘制训练结果
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, 'training_testing_plots.png')

