# autoencoder.py

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os

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
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# 微调模型（Fine-tuning）
def fine_tune_autoencoder(autoencoder, train_images, fine_tune_epochs=10):
    # 冻结编码器部分，只微调解码器部分
    for layer in autoencoder.layers[:4]:  # 假设前4层为编码器部分
        layer.trainable = False
    
    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    # 微调模型
    autoencoder.fit(train_images, train_images, epochs=fine_tune_epochs, batch_size=32, shuffle=True)

# 加载训练数据
def load_images(image_dir, target_size=(128, 128)):
    images = []
    for file_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
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

# 示例使用
input_image_path = 'path_to_input_image.jpg'
output_image_path = 'path_to_output_image.jpg'
train_images_path = 'path_to_training_images/'

# 加载并处理训练图像
train_images = load_images(train_images_path)

# 构建自动编码器
autoencoder_model = build_autoencoder((128, 128, 3))

# 假设我们有预训练模型的权重
pretrained_weights_path = 'pretrained_autoencoder.h5'
autoencoder_model.load_weights(pretrained_weights_path)

# 微调模型
fine_tune_autoencoder(autoencoder_model, train_images, fine_tune_epochs=10)

# 使用微调后的模型进行压缩和保存
compress_and_save(input_image_path, output_image_path, autoencoder_model)
