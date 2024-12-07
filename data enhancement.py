import os

import torch
from PIL import Image
import torchvision.transforms as transforms

# 设置路径
source_folder = './2 classified_train_merge_mixed'  # 替换为原始数据集路径
output_folder = './4 argumentation_train2'  # 替换为增强后数据集的保存路径

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义几何变换组合（如旋转、翻转、剪切）
# 定义几何变换组合（如旋转、翻转、剪切）
geometric_transforms = transforms.Compose([
    transforms.RandomRotation(45),                  # 随机旋转 ±45 度
    transforms.RandomHorizontalFlip(),              # 随机水平翻转
    transforms.RandomVerticalFlip(),                # 随机垂直翻转
    transforms.RandomAffine(degrees=0, shear=15)    # 随机剪切变换
])

# 定义尺度变换组合（如缩放和平移）
scaling_transforms = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),         # 随机缩放并裁剪至 64x64
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))    # 随机平移图像
])

# 定义颜色变换组合（如亮度、对比度、色调等）
color_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # 调整图像亮度、对比度、饱和度和色调
])

# 定义噪声添加组合（需要先转换为张量）
noise_transforms = transforms.Compose([
    transforms.ToTensor(),                                  # 将图像转换为张量（Tensor）
    transforms.GaussianBlur(kernel_size=3),                 # 添加高斯模糊，模拟噪声
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),     # 随机擦除部分图像
    transforms.ToPILImage()                                 # 将张量转换回 PIL 图像，用于保存
])

# 定义综合变换组合（结合多种增强方法）
combined_transforms = transforms.Compose([
    transforms.RandomRotation(20),                              # 随机旋转 ±20 度
    transforms.RandomHorizontalFlip(),                          # 随机水平翻转
    transforms.ColorJitter(brightness=0.3, contrast=0.3),       # 颜色调整
    transforms.RandomAffine(degrees=0, shear=10),               # 随机剪切
    transforms.RandomResizedCrop(64, scale=(0.9, 1.1))          # 随机缩放并裁剪
])

# origin+argumentation
enhance_round = 1
mix_enhance_round = 1


# 遍历源文件夹中的所有类别（例如：apple、banana 等）
for category in os.listdir(source_folder):
    category_path = os.path.join(source_folder, category)  # 获取每个类别的路径

    # 确保当前路径是文件夹
    if not os.path.isdir(category_path):
        continue

    # 创建对应类别的输出文件夹
    output_category_path = os.path.join(output_folder, category)
    if not os.path.exists(output_category_path):
        os.makedirs(output_category_path)

    # 遍历每个类别中的所有图像文件
    for filename in os.listdir(category_path):
        file_path = os.path.join(category_path, filename)  # 获取图像文件的完整路径
        print(filename)

        # 确保处理的是图像文件
        # 混合水果
        if filename.startswith('mixed') and filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_path).convert('RGB')  # 打开图像并转换为 RGB 格式
            # 保存原始图像到输出文件夹
            image.save(os.path.join(output_category_path, filename))
            # 应用几何变换多次
            for i in range(mix_enhance_round):
                geom_image = geometric_transforms(image)  # 应用几何变换
                geom_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_geom{i}.')}"))
            # 应用尺度变换多次
            for i in range(mix_enhance_round):
                scale_image = scaling_transforms(image)  # 应用尺度变换
                scale_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_scale{i}.')}"))
            # 应用颜色变换多次
            for i in range(mix_enhance_round):
                color_image = color_transforms(image)  # 应用颜色变换
                color_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_color{i}.')}"))
            # 应用噪声添加并保存多次（需要转换为张量）
            for i in range(mix_enhance_round):
                noise_image = noise_transforms(image)  # 应用噪声变换
                noise_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_noise{i}.')}"))
            # 应用综合变换并保存多次
            for i in range(mix_enhance_round):
                combined_image = combined_transforms(image)  # 应用综合变换
                combined_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_combined{i}.')}"))
        # 单个水果
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_path).convert('RGB')  # 打开图像并转换为 RGB 格式
            # 保存原始图像到输出文件夹
            image.save(os.path.join(output_category_path, filename))
            # 应用几何变换多次
            for i in range(enhance_round):
                geom_image = geometric_transforms(image)  # 应用几何变换
                geom_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_geom{i}.')}"))
            # 应用尺度变换多次
            for i in range(enhance_round):
                scale_image = scaling_transforms(image)  # 应用尺度变换
                scale_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_scale{i}.')}"))
            # 应用颜色变换多次
            for i in range(enhance_round):
                color_image = color_transforms(image)  # 应用颜色变换
                color_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_color{i}.')}"))
            # 应用噪声添加并保存多次（需要转换为张量）
            for i in range(enhance_round):
                noise_image = noise_transforms(image)  # 应用噪声变换
                noise_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_noise{i}.')}"))
            # 应用综合变换并保存多次
            for i in range(enhance_round):
                combined_image = combined_transforms(image)  # 应用综合变换
                combined_image.save(os.path.join(output_category_path, f"{filename.replace('.', f'_combined{i}.')}"))

print("图像增强完成，增强后的数据已保存！")