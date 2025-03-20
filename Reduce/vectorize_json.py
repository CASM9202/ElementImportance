import cv2
import numpy as np
import json
import os

from skimage.morphology import medial_axis
from skimage.measure import label, regionprops

# 类别定义
LABELS = {
    0: "Background",
    1: "Water",
    2: "Building_No_Damage",
    3: "Building_Minor_Damage",
    4: "Building_Major_Damage",
    5: "Building_Total_Destruction",
    6: "Vehicle",
    7: "Road-Clear",
    8: "Road-Blocked",
    9: "Tree",
    10: "Pool"
}

# 需要矢量化的类型映射
POINT_CLASSES = {6, 9, 10}  # Vehicle, Tree, Pool
POLYLINE_CLASSES = {7, 8}   # Road-Clear, Road-Blocked
POLYGON_CLASSES = {1, 2, 3, 4, 5}  # Water, Buildings


def smooth_curve(coords, epsilon):
    """
    使用 cv2.approxPolyDP 平滑曲线，去除尖角
    :param coords: 输入的坐标点列表
    :param epsilon: 逼近精度，值越大，平滑效果越强
    :return: 平滑后的坐标列表
    """
    # 将坐标点转为适合 OpenCV 的格式
    contour = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

    # 使用 cv2.approxPolyDP 进行折线逼近来平滑曲线
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 将逼近后的结果转换回列表格式
    return approx.reshape(-1, 2).tolist()
def extract_skeleton(mask):
    # """ 计算二值掩码的骨架，并转换为折线坐标 """
    # skeleton = medial_axis(mask)  # 计算骨架
    # labeled = label(skeleton)  # 标记骨架中的连通分量
    # polylines = []
    #
    # for region in regionprops(labeled):
    #     coords = region.coords  # 获取骨架点
    #     if len(coords) > 2:
    #         polylines.append(coords.tolist())  # 转换为列表
    #
    # return polylines
# 读取 PNG 并处理
    """ 计算二值掩码的骨架，并转换为折线坐标，平滑骨架 """
    # 计算骨架
    skeleton = medial_axis(mask)  # 计算骨架

    # 将骨架转为 uint8 类型，便于形态学操作
    skeleton = skeleton.astype(np.uint8)

    # 标记骨架中的连通分量
    labeled = label(skeleton)  # 连通区域标记

    polylines = []
    for region in regionprops(labeled):
        coords = region.coords  # 获取骨架点
        if len(coords) > 2:
            # 使用平滑曲线来处理坐标
            smoothed_coords = smooth_curve(coords, epsilon=800)
            polylines.append(smoothed_coords)  # 添加平滑后的坐标

    return polylines
def process_label_image(label_path):
    image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"无法读取 {label_path}")
        return None

    height, width = image.shape
    vector_data = []

    for class_id in np.unique(image):
        if class_id == 0:  # 背景不处理
            continue
        
        mask = (image == class_id).astype(np.uint8)  # 创建二值掩码
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # 过滤掉过小的对象
                continue
            if class_id in POLYLINE_CLASSES:
                # # 计算折线
                # epsilon = 0.01 * cv2.arcLength(contour, True)  # 近似折线
                # approx = cv2.approxPolyDP(contour, epsilon, False)
                # polyline = [[int(point[0][0]), int(point[0][1])] for point in approx]
                # # vector_data.append({"type": "Polyline", "category": LABELS[class_id], "coordinates": polyline})
                #
                # # 确保多边形是封闭的
                # if polyline and polyline[0] != polyline[-1]:
                #     polyline.append(polyline[0])  # 添加首尾连接点
                # # 将中心线坐标添加到数据中
                # vector_data.append({
                #     "type": "Polyline",
                #     "category": LABELS[class_id],
                #     "coordinates": polyline
                # })
                # 计算骨架
                skeleton_lines = extract_skeleton(mask)
                for skeleton in skeleton_lines:
                    polyline = [[int(x), int(y)] for y, x in skeleton]  # 变换坐标格式
                    vector_data.append({
                        "type": "Polyline",
                        "category": LABELS[class_id],
                        "coordinates": polyline
                    })




            elif class_id in POINT_CLASSES:
                # 计算质心，转换为点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    vector_data.append({"type": "Point", "category": LABELS[class_id], "coordinates": [cx, cy]})

            elif class_id in POLYGON_CLASSES:
                # 转换为多边形
                polygon = [[int(point[0][0]), int(point[0][1])] for point in contour]
                vector_data.append({"type": "Polygon", "category": LABELS[class_id], "coordinates": [polygon]})



    return vector_data


# 处理所有 PNG
label_dir = os.path.abspath("../Data/TestData/label")
output_path = os.path.abspath("output/vectors.json")

all_vector_data = []

for filename in os.listdir(label_dir):
    if filename.endswith(".png"):
        label_path = os.path.join(label_dir, filename)
        vectors = process_label_image(label_path)
        if vectors:
            all_vector_data.append({"file": filename, "features": vectors})

# 写入 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_vector_data, f, indent=4)

print(f"矢量化完成，结果已保存至 {output_path}")
