import os
import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from realesrgan import RealESRGAN
from skimage import img_as_float
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor, ToPILImage
import logging

class ImageEnhancer:
    def __init__(self, model_name='RealESRGAN_x4plus', weights_path='RealESRGAN_x4plus.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 创建一个ESRGAN模型实例
        self.model = RealESRGAN(model_name=model_name)  # 使用4倍放大模型
        self.model.load_weights(weights_path)  # 确保这个路径有预训练权重 
        self.model.to(self.device)  # 移动模型到指定设备
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.to_tensor = ToTensor()
        self.to_pil_image = ToPILImage()

    def upscale_image(self, input_path, output_path):
        try:
            # 读取图像
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Image cannot be read, possibly due to invalid file path or file format.")
            # 转换图像为tensor，并移动到GPU
            img_tensor = self.to_tensor(img).unsqueeze(0).to(self.device)  # 使用torchvision的to_tensor自动处理归一化和维度
            # 使用Real-ESRGAN增强图像
            output_tensor, _ = self.model.predict(img_tensor)
            # 将输出移回CPU，并转换为PIL Image以保存为文件
            output_image = self.to_pil_image(output_tensor.squeeze(0).cpu())
            output_image.save(output_path)
            logging.info(f"Processed {input_path} successfully.")
        except Exception as e:
            logging.error(f"Failed to process {input_path}: {str(e)}")

    def upscale_images_in_folder(self, folder_path, output_folder, max_workers=4):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(self.supported_formats):
                    input_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_folder, filename)
                    executor.submit(self.upscale_image, input_path, output_path)

    def evaluate_image_quality(self, original_img_path, enhanced_img_path):
        orig_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
        enh_img = cv2.imread(enhanced_img_path, cv2.IMREAD_GRAYSCALE)
        orig_img = img_as_float(orig_img)
        enh_img = img_as_float(enh_img)
        psnr_value = psnr(orig_img, enh_img)
        ssim_value = ssim(orig_img, enh_img, data_range=enh_img.max() - enh_img.min())
        return psnr_value, ssim_value

    def load_and_enhance_ui(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            enhanced_path = 'enhanced_image.jpg'
            self.upscale_image(file_path, enhanced_path)
            img = Image.open(enhanced_path)
            img = ImageTk.PhotoImage(img)
            panel = tk.Label(self.window, image=img)
            panel.image = img
            panel.pack()
            # 图像质量评估
            psnr_value, ssim_value = self.evaluate_image_quality(file_path, enhanced_path)
            quality_label = tk.Label(self.window, text=f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")
            quality_label.pack()

    def setup_ui(self):
        window = tk.Tk()
        load_button = tk.Button(self.window, text="Load Image", command=self.load_and_enhance_ui)
        load_button.pack()
        self.window.mainloop()

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 示例使用
enhancer = ImageEnhancer()
enhancer.setup_ui()
enhancer.upscale_images_in_folder('path_to_input_folder', 'path_to_output_folder')
