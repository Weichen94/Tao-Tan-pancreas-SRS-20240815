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
    Image.MAX_IMAGE_PIXELS = None  # 或者任何你期望的数值
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=3).to(device)
    model_weight_path = "E:\code\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\pytorch_classification\Test11_efficientnetV2\weights\model-19.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    target_layers = [model.blocks[2]]


    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "l"

    data_transform = transforms.Compose(
        [#transforms.Resize(img_size[num_model][1]),
         #transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "21.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    original_img = Image.open(img_path).convert('RGB')
    original_size=original_img.size
    crop_size=(384,480)
    stride=200
    result_img = np.array(original_img, dtype=np.float32) / 255.0
    # img = center_crop_img(img, 224)
    for y in range(0, original_size[1] - crop_size[1] + 1, stride):
        for x in range(0, original_size[0] - crop_size[0] + 1, stride):
            # 裁剪图像
            cropped_img = original_img.crop((x, y, x + crop_size[0], y + crop_size[1]))

            # 转换为 NumPy 数组
            cropped_array = np.array(cropped_img, dtype=np.uint8)

            # [C, H, W]
            img_tensor = data_transform(cropped_array)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            target_category = 0  # tabby, tabby cat

            # 获取 Grad-CAM 输出
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            # 将 Grad-CAM 输出调整为裁剪图像大小
            resized_cam = cv2.resize(grayscale_cam[0], (crop_size[0], crop_size[1]))

            # 复制原始图像
            result_img_copy = copy.deepcopy(result_img)

            # 将 Grad-CAM 叠加到复制的图像上
            result_img[y:y + crop_size[1], x:x + crop_size[0], :] += resized_cam[:, :, np.newaxis]

    # 对结果图像进行 min-max 归一化
    result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
    # 调整图像亮度
    brightness_factor = 2  # 调整亮度的因子，可以根据需要调整
    result_img = result_img * brightness_factor

    # 将亮度调整后的图像限制在 [0, 255] 范围内
    result_img = np.clip(result_img * 255, 0, 255).astype(np.uint8)

    # 输出热力图
    plt.imshow(result_img[:, :, 0], cmap='coolwarm', interpolation='nearest')

    # 保存热力图
    heatmap_output_path = "heatmap_result5300.png"
    plt.axis('off')
    plt.savefig(heatmap_output_path,dpi=1200)

    # 显示热力图
    plt.show()

if __name__ == '__main__':
    main()
