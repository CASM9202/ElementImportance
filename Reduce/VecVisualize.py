import json
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

# 定义文件路径
vector_file_path = "output/vectors.json"
output_dir = "output"
color_text_file = r"..\Data\QGIS_label_style.txt"
image_dir = r"..\Data\TestData\org"  # 原始图像目录路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取颜色文本文件，获取颜色映射
def load_color_map(file_path):
    color_map = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("INTERPOLATION:"):
                parts = line.split(",")
                if len(parts) >= 6:
                    category = parts[-1].strip()
                    rgb = [int(part) for part in parts[1:3]]
                    alpha = int(parts[3])
                    color = "#{:02X}{:02X}{:02X}".format(*rgb, alpha)
                    color_map[category] = color
    return color_map

# 读取矢量数据
def load_vectors(file_path):
    with open(file_path, "r") as f:
        vectors = json.load(f)
    return vectors

def draw_polyline(ax, coordinates, color, label):
    """
    绘制折线
    :param ax: matplotlib轴对象
    :param coordinates: 折线的坐标列表，格式为 [[x1, y1], [x2, y2], ...]
    :param color: 折线的颜色
    :param label: 折线的标签
    """
    xs, ys = zip(*coordinates)  # 解包坐标点
    ax.plot(xs, ys, color=color, linewidth=2, label=label)


# 绘制矢量图
def draw_vectors_on_image(image_path, vectors, color_map, output_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        return

    # # 加载原始图像
    # img = np.array(Image.open(image_path))
    # img_width, img_height = img.shape[1], img.shape[0]
    # fig, ax = plt.subplots()
    # ax.imshow(img)  # 显示变暗的图像
    # ax.axis('off')  # 关闭坐标轴
    # 加载原始图像
    img = np.array(Image.open(image_path))
    img_width, img_height = img.shape[1], img.shape[0]

    # 将图像亮度降低到原来的一定比例
    brightness_factor = 0.5  # 这里的0.5表示亮度变为原来的50%
    img_darker = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
    # 垂直翻转图像
    img_darker = np.flipud(img_darker)


    fig, ax = plt.subplots()
    ax.imshow(img_darker)  # 显示变暗的图像
    ax.axis('off')  # 关闭坐标轴

    # 遍历features，绘制每种几何类型
    for feature in vectors['features']:
        feature_type = feature["type"]
        category = feature["category"]
        coordinates = feature["coordinates"]
        color = color_map.get(category, "white")  # 根据category获取颜色

        if feature_type == "Polygon":
            # 处理多边形
            # 调整坐标系，将y坐标反转
            adjusted_coordinates = [(x, img_height - y) for x, y in coordinates[0]]
            poly = Polygon(adjusted_coordinates, closed=True, edgecolor=color, facecolor='none', linewidth=2, alpha=1)
            ax.add_patch(poly)
        elif feature_type == "Polyline":
            # 调整坐标系，将y坐标反转
            adjusted_coordinates = [(x, img_height - y) for x, y in coordinates]
            xs, ys = zip(*adjusted_coordinates)  # 解包坐标点
            ax.plot(xs, ys, color=color, linewidth=2)  # 绘制折线
        elif feature_type == "Point":
            if len(coordinates) > 1:  # 确保坐标点数量正确
                x, y = coordinates  # 直接解包坐标点
                # 调整坐标系，将y坐标反转
                ax.plot(x, img_height - y, marker="o", color=color, markersize=10)  # 绘制点

    # 设置坐标范围
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)

    # 保存图像
    plt.savefig(output_path, transparent=True)
    plt.close(fig)

# 主程序
def main():
    # 加载颜色映射
    color_map = load_color_map(color_text_file)
    # print("Loaded color map:", color_map)

    # 加载矢量数据
    vectors = load_vectors(vector_file_path)
    # print("Loaded vectors:", vectors)

    # 遍历矢量数据并绘制图像
    for item in vectors:
        file_name = item["file"]  # 获取文件名
        file_name = file_name.replace("_lab", "")  # 移除文件名中的 "_lab"
        image_file_extension = os.path.splitext(file_name)[1]  # 获取文件扩展名
        file_name = os.path.splitext(file_name)[0] + ".jpg"  # 修改文件扩展名为.jpg
        features = item["features"]
        image_path = os.path.join(image_dir, file_name)  # 获取图像路径
        output_path = os.path.join(output_dir, f"{file_name.split('.')[0]}_vec.png")  # 输出文件名

        # 绘制矢量化结果在图像上
        draw_vectors_on_image(image_path, {"features": features}, color_map, output_path)
        print(f"Generated image for file {file_name}: {output_path}")

if __name__ == "__main__":
    main()