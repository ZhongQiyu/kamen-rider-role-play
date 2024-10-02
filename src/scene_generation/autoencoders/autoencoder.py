# autoencoder.py

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 加载图像并获取其尺寸
def load_image_size(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像 {image_path}")
    img_shape = img.shape  # 获取图像尺寸 (height, width, channels)
    return img_shape

# 简单的图像压缩函数 (通过缩放和降低质量)
def compress_image(img, scale_percent=50):
    # 获取原图像的尺寸
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # 使用OpenCV的resize函数进行图像压缩
    compressed_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return compressed_img

# 简单的图像解压函数 (通过放大)
def decompress_image(img, original_size):
    # 使用OpenCV的resize函数进行图像解压
    decompressed_img = cv2.resize(img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    return decompressed_img

# 压缩和解压缩的流程
def compress_and_save(input_path, output_path, scale_percent=50):
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法加载图像 {input_path}")
    print(f"Input image loaded with shape: {img.shape}")

    # 压缩图像
    compressed_img = compress_image(img, scale_percent)
    print(f"Compressed image shape: {compressed_img.shape}")

    # 解压缩图像
    decompressed_img = decompress_image(compressed_img, img.shape)
    print(f"Decompressed image shape: {decompressed_img.shape}")

    # 将输出像素值调整回 [0, 255] 并保存解压后的图像
    decompressed_img = np.clip(decompressed_img, 0, 255).astype('uint8')
    cv2.imwrite(output_path, decompressed_img)
    print(f"Decompressed image saved to: {output_path}")

# 创建示例训练图片
def create_sample_images(num_images=5, image_size=(128, 128), output_dir='sample_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_images):
        img = Image.new('RGB', image_size, color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        # Drawing random shapes on the image
        for _ in range(5):
            x0, y0 = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
            x1, y1 = np.random.randint(x0, image_size[0]), np.random.randint(y0, image_size[1])
            shape_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
        img_path = os.path.join(output_dir, f"image_{i+1}.jpg")
        img.save(img_path)
    print(f"{num_images} images saved to '{output_dir}'")

# 示例使用
input_image_path = 'passport_bio.jpg'
output_image_path = 'passport_bio_compressed.jpg'
train_images_path = 'sample_images/'  # 使用生成的示例图片

# 生成训练图片
create_sample_images(num_images=10, output_dir=train_images_path)

# 获取输入图像的大小
input_image_size = load_image_size(input_image_path)

# 压缩和保存图像
compress_and_save(input_image_path, output_image_path, scale_percent=50)
