import os
import shutil
from collections import defaultdict

# 设置源文件夹路径和目标文件夹路径
source_folder = './3 picture_merger/train_white'  # 替换为你的图片所在的文件夹路径
destination_folder = './3 picture_merger/classified_train_white'  # 替换为目标分类文件夹路径

# 确保目标文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 创建一个字典来存储每个类别的图片数量
category_count = defaultdict(int)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 确保只处理图片文件
        # 获取类别名称（根据下划线分割字符串的第一个部分）
        category = filename.split('_')[0]

        # 创建类别文件夹
        category_folder = os.path.join(destination_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # 复制文件到对应的类别文件夹
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(category_folder, filename)
        shutil.copy2(source_path, destination_path)     # 使用 copy2 保留文件的元数据

        # 更新类别计数
        category_count[category] += 1

# 打印每个类别的图片数量
print("图片分类完成！\n每个类别的图片数量如下：")
for category, count in category_count.items():
    print(f"{category}: {count} 张图片")
