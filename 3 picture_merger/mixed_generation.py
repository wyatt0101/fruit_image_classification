import os
import random
from PIL import Image

# 定义分类文件夹路径和输出文件夹
base_path = 'classified_train_white'
output_path = 'merged_images'
categories = ['apple', 'banana', 'orange']
target_size = (200, 200)  # 统一目标尺寸（宽, 高）


# 随机选择指定数量的类别，并从中选取图片
def select_images(base_path, categories, num_images):
    selected_images = []
    selected_categories = random.sample(categories, k=num_images)  # 确保选择指定数量的类别
    for category in selected_categories:
        category_path = os.path.join(base_path, category)
        images = os.listdir(category_path)
        if not images:
            raise ValueError(f"文件夹 {category_path} 为空，请确保每个类别文件夹中都有图片。")
        img_name = random.choice(images)  # 随机选择图片
        selected_images.append(os.path.join(category_path, img_name))
    return selected_images


# 调整图片大小
def resize_image(img, target_size):
    """
    将图片调整为目标尺寸。
    使用填充方式保持原始比例，避免拉伸变形。
    """
    img = img.convert('RGB')  # 确保图片模式一致
    img.thumbnail(target_size, Image.ANTIALIAS)  # 等比例缩放，适配目标尺寸
    background = Image.new('RGB', target_size, (255, 255, 255))  # 白色背景
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    background.paste(img, (paste_x, paste_y))  # 将图片居中粘贴
    return background


# 合并图片
def merge_images(images):
    # 打开并调整图片大小
    img_list = [resize_image(Image.open(img), target_size) for img in images]

    # 获取所有图片的宽高
    widths, heights = zip(*(img.size for img in img_list))

    # 创建空白画布，背景色为白色
    canvas_width = max(widths) * 2
    canvas_height = max(heights) * 2
    new_img = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    if len(img_list) == 2:  # 两张图片
        new_img.paste(img_list[0], (0, 0))
        new_img.paste(img_list[1], (widths[0], 0))
    elif len(img_list) == 3:  # 三张图片
        mode = random.randint(0, 1)  # 随机选择正三角或倒三角
        if mode == 0:  # 倒三角
            new_img.paste(img_list[0], (0, 0))
            new_img.paste(img_list[1], (widths[0], 0))
            new_img.paste(img_list[2], (int(widths[0] - widths[2] / 2), max(heights[0], heights[1])))
        elif mode == 1:  # 正三角
            new_img.paste(img_list[0], (int(widths[1] / 2), 0))
            new_img.paste(img_list[1], (0, heights[0]))
            new_img.paste(img_list[2], (widths[1], heights[0]))

    return new_img


# 主逻辑
def main():
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_images = 50  # 设置需要生成的图片总数
    num_two_images = num_images // 2  # 两张图片的数量
    num_three_images = num_images - num_two_images  # 三张图片的数量

    counter_two = 0
    counter_three = 0

    for i in range(num_images):
        if counter_two < num_two_images:
            images = select_images(base_path, categories, 2)  # 随机选择两张图片
            counter_two += 1
        elif counter_three < num_three_images:
            images = select_images(base_path, categories, 3)  # 随机选择三张图片
            counter_three += 1

        # 合并图片
        result_img = merge_images(images)

        # 保存合成的图片，命名为 result_1.jpg, result_2.jpg 等
        output_file = os.path.join(output_path, f'result_{i + 1}.jpg')
        result_img.save(output_file)
        print(f"第 {i + 1} 张图片合并完成，保存为 {output_file}")


if __name__ == "__main__":
    main()
