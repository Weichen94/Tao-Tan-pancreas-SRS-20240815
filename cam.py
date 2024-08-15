import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model import efficientnetv2_l as create_model
import cv2
import copy
def main():
    Image.MAX_IMAGE_PIXELS = None  # 设置图像最大像素数，避免出现警告
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = create_model(num_classes=2).to(device)
    model_weight_path = r"0630weights0.985\995-947 good.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 设置目标图层
    # target_layers = [model.features[-1]]
    target_layers = [model.blocks[-1]]

    # 定义数据转换
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载图像
    img_path = "0701-2.png"
    assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
    original_img = Image.open(img_path).convert('RGB')
    original_size = original_img.size

    # 定义裁剪大小和步长
    crop_size = (384, 480)
    stride = 200

    # 初始化结果图像
    result_img = np.array(original_img, dtype=np.float32) / 255.0

    # 使用滑动窗口的方式对图像进行裁剪和分析
    for y in range(0, original_size[1] - crop_size[1] + 1, stride):
        for x in range(0, original_size[0] - crop_size[0] + 1, stride):
            # 裁剪图像
            cropped_img = original_img.crop((x, y, x + crop_size[0], y + crop_size[1]))

            # 转换为 NumPy 数组并应用数据转换
            cropped_array = np.array(cropped_img, dtype=np.uint8)
            img_tensor = data_transform(cropped_array)

            # 添加批次维度并移到 GPU 上
            input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            # 创建 GradCAM 实例并获取 Grad-CAM 输出
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            target_category = 0  # 定义目标类别
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            # 将 Grad-CAM 输出调整为裁剪图像大小
            resized_cam = cv2.resize(grayscale_cam[0], (crop_size[0], crop_size[1]))

            # 将 Grad-CAM 叠加到原始图像上
            result_img[y:y + crop_size[1], x:x + crop_size[0], :] += resized_cam[:, :, np.newaxis]

    # 对结果图像进行 min-max 归一化并调整亮度
    result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
    brightness_factor = 2  # 调整亮度的因子
    result_img = result_img * brightness_factor
    result_img = np.clip(result_img * 255, 0, 255).astype(np.uint8)

    # 输出热力图并保存
    heatmap_output_path = "heatmap0701.png"
    original_img_np = np.array(original_img, dtype=np.float32) / 255.
    visualization = show_cam_on_image(original_img_np.astype(dtype=np.float32) / 255., result_img, use_rgb=True)
    plt.imsave(heatmap_output_path, result_img)
    plt.imshow(visualization)
    plt.axis('off')
    plt.show()

# 调用主函数
main()