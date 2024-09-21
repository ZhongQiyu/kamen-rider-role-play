# autoencoder.py

import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# 自动编码器模型的构建
def build_autoencoder(input_shape):
    # 编码器
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoder_output = layers.MaxPooling2D((2, 2), padding='same')(x)

    # 解码器
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # 自动编码器模型
    autoencoder = models.Model(encoder_input, decoder_output)
    return autoencoder

# 感知损失函数
def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg.trainable = False
    # 使用 VGG16 提取特征
    true_features = vgg(y_true)
    pred_features = vgg(y_pred)
    return tf.reduce_mean(tf.square(true_features - pred_features))

# 微调模型
def fine_tune_autoencoder(autoencoder, train_images, fine_tune_epochs=10):
    if train_images is not None and len(train_images) > 0:
        # 冻结编码器部分，只微调解码器部分
        for layer in autoencoder.layers[:4]:  # 假设前4层为编码器部分
            layer.trainable = False
        
        # 编译模型
        autoencoder.compile(optimizer='adam', loss=perceptual_loss)

        # 微调模型
        autoencoder.fit(train_images, train_images, epochs=fine_tune_epochs, batch_size=32, shuffle=True)
    else:
        print("没有训练数据集，跳过微调过程。")

# 加载训练数据
def load_images(image_dir, target_size=(128, 128)):
    images = []
    if os.path.exists(image_dir):
        for file_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            images.append(img)
    else:
        print(f"目录 {image_dir} 不存在，无法加载训练数据。")
    return np.array(images)

# 压缩和解压缩的流程
def compress_and_save(input_path, output_path, model):
    # 读取并调整图像大小
    img = cv2.imread(input_path)
    img = cv2.resize(img, (128, 128))  # 假设图像大小为128x128
    img = img.astype('float32') / 255.0

    # 预测（压缩和解压缩）
    img = np.expand_dims(img, axis=0)
    compressed_img = model.predict(img)

    # 保存压缩后的图像
    compressed_img = np.squeeze(compressed_img) * 255
    compressed_img = compressed_img.astype('uint8')
    cv2.imwrite(output_path, compressed_img)

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
output_image_path = 'patssport_bio_compressed.jpg'
train_images_path = 'sample_images/'  # 使用生成的示例图片

# 生成训练图片
create_sample_images(num_images=10, output_dir=train_images_path)

# 加载并处理训练图像
train_images = load_images(train_images_path)

# 构建自动编码器
autoencoder_model = build_autoencoder((128, 128, 3))

# 假设我们有预训练模型的权重
# pretrained_weights_path = 'pretrained_autoencoder.h5'
# if os.path.exists(pretrained_weights_path):
#     autoencoder_model.load_weights(pretrained_weights_path)
# else:
#     print(f"预训练权重文件 {pretrained_weights_path} 不存在，使用随机初始化的权重。")

# 微调模型
# fine_tune_autoencoder(autoencoder_model, train_images, fine_tune_epochs=10)

# 使用微调后的模型进行压缩和保存
compress_and_save(input_image_path, output_image_path, autoencoder_model)
