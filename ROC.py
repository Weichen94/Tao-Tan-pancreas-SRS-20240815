import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['axes.unicode_minus'] = False  # 处理图像显示负号的问题
import seaborn as sns

sns.set(font="Kaiti", style="ticks", font_scale=1.4)  # 设置seaborn的字体和样式
from sklearn.metrics import roc_curve, auc

# 指定txt文件的路径
file_path = r'test-cancer.txt'  # 替换为实际的txt文件路径
file_path_neg = r'test-normal.txt'  # 替换为实际的包含true值为0的txt文件路径
# 创建一个空列表来存储提取的内容
extracted_data = []

# 打开文件并读取内容
with open(file_path, "r") as file:
    for line in file:
        # 检查行中是否包含"prob:"
        if "prob:" in line:
            # 使用split方法根据"prob:"分割字符串，并取分割后的第二部分（索引为1）
            # 注意：split会返回一个列表，所以我们需要索引来获取具体的元素
            # strip()用于去除可能存在的换行符或其他空白字符
            extracted_part = line.split("prob:")[1].strip()
            # 将提取的部分添加到列表中
            extracted_data.append(extracted_part)

# 将提取的内容转换为numpy数组，并转换为浮点数类型
y_pred = np.array(extracted_data, dtype=float)

# 创建一个与y_pred长度相同的全为1的y_true数组
y_true = np.ones_like(y_pred)

# 打开包含true值为0的预测样本文件并读取内容
with open(file_path_neg, "r") as file:
    for line in file:
        # 检查行中是否包含"prob:"
        if "prob:" in line:
            # 使用split方法根据"prob:"分割字符串，并取分割后的第二部分（索引为1）
            extracted_part = line.split("prob:")[1].strip()
            # 将提取的部分转换为浮点数并计算1减去该值
            extracted_part = 1.0 - float(extracted_part)
            # 将计算结果添加到extracted_data列表中
            extracted_data.append(extracted_part)

# 将新的extracted_data转换为numpy数组，并转换为浮点数类型
y_pred = np.array(extracted_data, dtype=float)

# 计算添加的样本数量
num_added_samples = len(extracted_data) - len(y_true)

# 在y_true后面添加相应数量的0
y_true = np.concatenate([y_true, np.zeros(num_added_samples)])

print("y_pred:", y_pred)
print("y_true:", y_true)

# 计算ROC曲线的FPR和TPR
fpr_Nb, tpr_Nb, _ = roc_curve(y_true, y_pred)
# 计算AUC值
aucval = auc(fpr_Nb, tpr_Nb)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线作为参考
plt.plot(fpr_Nb, tpr_Nb, "r", linewidth=3)  # 绘制ROC曲线
plt.grid(False)  # 显示网格
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.title("ROC曲线")
plt.text(0.15, 0.9, "AUC = " + str(round(aucval, 4)), fontsize=14)  # 显示AUC值
plt.show()