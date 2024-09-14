# visualizer.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, BertForMaskedLM

class FeatureVisualizer:
    """特征可视化类，负责处理特征和文本的可视化。"""
    
    def __init__(self, model_name="cl-tohoku/bert-base-japanese"):
        """初始化可视化器，并加载所需的模型和tokenizer。"""
        self.correction_model, self.correction_tokenizer = self.load_models(model_name)

    def load_models(self, model_name):
        """加载文本纠正模型和对应的tokenizer。"""
        correction_model = BertForMaskedLM.from_pretrained(model_name)
        correction_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return correction_model, correction_tokenizer
    
    def plot_feature_vector(self, feature_vector, title="Feature Vector Visualization"):
        """可视化特征向量。"""
        plt.figure(figsize=(10, 5))
        sns.heatmap(feature_vector.cpu().numpy(), annot=True, fmt=".2f", cmap="viridis")
        plt.title(title)
        plt.show()

    def plot_text_correction(self, original_text, corrected_text):
        """可视化文本纠正前后的对比。"""
        print("Original Text: ", original_text)
        print("Corrected Text: ", corrected_text)
        # 此处可以拓展为更高级的对比可视化

    def correct_text(self, text, device='cpu'):
        """对输入文本进行纠正，并返回纠正后的文本。"""
        masked_text = text.replace("間違った", "[MASK]")
        encoded_input = self.correction_tokenizer(masked_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.correction_model(**encoded_input)
            predictions = outputs.logits
        
        masked_index = torch.where(encoded_input["input_ids"] == self.correction_tokenizer.mask_token_id)[1]
        predicted_id = predictions[0, masked_index].argmax(dim=-1)
        predicted_token = self.correction_tokenizer.decode(predicted_id).strip()
        
        corrected_text = masked_text.replace("[MASK]", predicted_token)
        
        self.plot_text_correction(text, corrected_text)
        return corrected_text

class MultiModalProcessor:
    """多模态处理器类，负责处理和整合文本、音频和图像特征。"""

    def __init__(self):
        """初始化多模态处理器。"""
        pass

    def process_text(self, text):
        """处理文本数据并返回特征张量。"""
        print(f"Processing text: {text}")
        return torch.randn((1, 768))  # 假设文本特征是768维的向量

    def process_audio(self, audio_path):
        """处理音频数据并返回特征张量。"""
        print(f"Processing audio: {audio_path}")
        return torch.randn((1, 768))  # 假设音频特征是768维的向量

    def process_image(self, image_path):
        """处理图像数据并返回特征张量。"""
        print(f"Processing image: {image_path}")
        return torch.randn((1, 768))  # 假设图像特征是768维的向量

    def combine_features(self, text_features, audio_features, image_features):
        """结合文本、音频和图像特征。"""
        combined_features = torch.cat([text_features, audio_features, image_features], dim=-1)
        return combined_features

class VisualizationHelper:
    """用于加载、清理和绘制数据的帮助类。"""
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.data = None

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """从CSV文件加载数据。"""
        if filepath:
            self.filepath = filepath
        if not self.filepath:
            raise ValueError("Filepath must be provided to load data.")
        
        self.data = pd.read_csv(self.filepath)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """执行基本的数据清理操作。"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.data = self.data.dropna()  # 删除包含缺失值的行
        self.data = self.data[self.data.select_dtypes(include=['number']).ge(0).all(1)]  # 删除包含负值的行
        return self.data

    def create_bar_plot(self, x_col: str, y_col: str, title: str) -> px.bar:
        """创建柱状图。"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.bar(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_line_plot(self, x_col: str, y_col: str, title: str) -> px.line:
        """创建折线图。"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.line(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_scatter_plot(self, x_col: str, y_col: str, title: str) -> px.scatter:
        """创建散点图。"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.scatter(self.data, x=x_col, y=y_col, title=title)
        return fig

class DataVisualizer:
    """数据可视化类，集成了加载、清理和绘图功能。"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.helper = VisualizationHelper(filepath)

    def load_and_clean_data(self):
        self.data = self.helper.load_data()
        self.data = self.helper.clean_data()
    
    def generate_plot(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_and_clean_data() first.")
        
        fig = self.helper.create_bar_plot(x_col="Category", y_col="Values", title="Category Values")
        return fig

class AgentVisualizer:
    """用于代理（agent）数据的可视化类。"""
    
    def __init__(self, data=None):
        if data is None:
            data = {
                "Agent": ["Agent1", "Agent2", "Agent3"],
                "Performance": [80, 90, 85]
            }
        self.df = pd.DataFrame(data)
        self.helper = VisualizationHelper()

    def set_data(self, data):
        self.df = pd.DataFrame(data)

    def get_data(self):
        return self.df

    def generate_plot(self, x_col="Agent", y_col="Performance", title="Agent Performance"):
        self.helper.data = self.df
        return self.helper.create_line_plot(x_col=x_col, y_col=y_col, title=title)

def main():
    # 示例1: 使用 FeatureVisualizer 进行文本纠正和特征可视化
    visualizer = FeatureVisualizer()
    text = "这里有一个間違った的例子"
    corrected_text = visualizer.correct_text(text)

    feature_vector = torch.randn((1, 768))
    visualizer.plot_feature_vector(feature_vector, title="Example Feature Vector")

    # 示例2: 使用 MultiModalProcessor 处理多模态数据
    processor = MultiModalProcessor()
    text_features = processor.process_text(corrected_text)
    audio_features = processor.process_audio('path_to_audio.wav')
    image_features = processor.process_image('path_to_image.jpg')

    combined_features = processor.combine_features(text_features, audio_features, image_features)
    visualizer.plot_feature_vector(combined_features, title="Combined Feature Vector")

    # 示例3: 使用 DataVisualizer 进行数据可视化
    data_visualizer = DataVisualizer('path_to_csv_file.csv')
    data_visualizer.load_and_clean_data()
    fig = data_visualizer.generate_plot()
    fig.show()

    # 示例4: 使用 AgentVisualizer 进行代理性能可视化
    agent_visualizer = AgentVisualizer()
    agent_plot = agent_visualizer.generate_plot()
    agent_plot.show()

if __name__ == "__main__":
    main()
