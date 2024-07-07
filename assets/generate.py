import cv2
import numpy as np

# 打开原图和mask图
original_image_path = 'stool.png'
mask_image_path = 'stool_mask.png'

# 使用cv2读取图像
original_image = cv2.imread(original_image_path)
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# # 调整mask图的大小
# mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_LINEAR)

# # 调整原图的大小
# original_image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_LINEAR)

# 创建一个全白图像
white_image = np.ones_like(original_image) * 255

# 生成masked image
masked_image = np.where(mask_image[:, :, None] >= 127, white_image, original_image)

# 保存结果图像
cv2.imwrite(mask_image_path.replace("mask.png","masked_image.png"), masked_image)
