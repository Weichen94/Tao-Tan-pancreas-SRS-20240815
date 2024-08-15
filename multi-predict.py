import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_l as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "l"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         ])
    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r"E:\dataset\new yixian\test Non-diagnostic"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]

    # load image
    # img_path = r"E:\dataset\new yixian\test\cancer\4.png"
    # assert os.path.exists(img_path_list), "file: '{}' dose not exist.".format(img_path_list)
    # img = Image.open(img_path_list)
    # plt.imshow(img)
    # [N, C, H, W]
    # img = data_transform(img)
    # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = r"weights/model-586.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    batch_size = 2
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    #
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()


if __name__ == '__main__':
    main()
