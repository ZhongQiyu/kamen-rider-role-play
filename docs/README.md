# kamen-rider-blade

これは「仮面ライダー剣」のマルチエージェントシステムです。

このプロジェクトは、質問応答を利用する再現可能なプロジェクトです。

必ずオリジナルの仕事になります。戦わない理由はありません。あなたの存在が、「最後の切り札」です。

「更新するの予告」
７月19日

https://www.acfun.cn/v/ac32066706
https://www.acfun.cn/v/ac32067906
https://www.acfun.cn/v/ac32433574

J-POP HTML
Visual Kei HTML
Music (Mainly Pop/Rock) HTML
Anime/Game CD & DVD HTML
Book HTML
Book TEXT
Figure/Character/Hobby/Toy
Plastic Models
Movies/TV Series HTML
Movies/TV Series TEXT
Makeup Brush & Japanese Craft
Audio (incl. Headphone & Cable)
Gravure Idol (Model)
New Release Updates (TEXT Only)
K-POP
Anime Collectibles (Figure/Character/Toy)
Anime/Voice Artist CD & DVD
Drama CD & Radio CD
Video Games & Game CD
Tokusatsu/Super Sentai
Apparel
Classical Music
Jazz
Vinyl



假面骑士剑 广播剧

请问有没有办法能够做一下你memory update的自动管理

就是我有时候会突然换话题问你，你要是觉得太突兀了就更新记忆的话跟我说下，确认更新记忆你再把它记下，可以吗

你能帮我量化一下这个 低 中 高 怎么做好一些吧 用具体的数值看看

请针对你已有的大模型的知识库 给我个例子吧
将上面你提到的这些维度逐一对比 尽可能量化数据 用表格做一下最好

如果我要对三个大模型对比 除了基本介绍 部署方法 性能评测
还应该对比什么 单说性能、速度和资源占用率这三点 应该是怎么去对应
请给出尽可能具体的建议等

能够说“对算法添加物理规则吗”？
有“算法和视频符合物理规律”这种说法吗？
降1倍是原数值1/2, 降2倍是原数值1/3对吧

我现在还剩5gb内存：

- 拉起本地逾7gb的日文原文wav实时音频转写和翻译流程
- 根据前5-10轮给出的这些概念解析，将上下义概念进行连接
- 使用半参量化和bf16混合精度推理，执行书生·浦语一代7B大模型的中文对话全流程
- 根据一个400kb的日语影视数据集，测试在ChatRTX中回调和检索生成中、日、英三语的风格迁移

现在请你从以上任务中选一个执行并写出完整的代码。

关于对话系统部署，我有几个问题想问：

- 工作流化的提示词迭代会更好些吗？
- 我现在卡在提示工程化和清洗数据集，和LangChain等nlp工作流工具怎么联系？
- 制作一个基于《假面骑士剑》台词训练的多智能体系统，训练硬件规格与成本估算是多少？
- 我搜集好的台词只有说话人名字、时间戳、内容, 是否应：
  - 表示情感的标签使用二元分类而不是回归值？
  - 在原数据中加入正负样本, 从而更好地调整llm的训练方式？
  - 加入中文和英文到日语的互译模块后，如何对训练文本的语义消歧？
  - 从台词的视角看，如何识别转场？假设每一场的每一幕构成最小滑动窗口，那么scene_id，timestamp_start，timestamp_end，和role_info应该如何提取？
  - 一个对话来回不是严格的a对b然后b回复a，如果要做原台词的检索生成，会对数据集的真实性造成影响吗？如果会，用什么微调trick；如果不会，怎么最大程度在日语上单个增强真实性？

什么样的记忆我应该是留着

"自动标注校正"和"手动校正"交给音频转写模型去做好吗
"数据扩展", "上下文重新构造"和"情感标注"能否归为一类？
"迭代增强和微调", "模型评估"和"数据增强的迭代"能否和策略结合？

我现在有一个基于《假面骑士剑》日语对话数据集，现在：

- 微调好了的t5模型，但是现在没有对话能力，需要实现文本微调
- 对embedding做下修改，分为position embedding和word embedding
- 对attention模块做下mask操作的修改，将两种embeddings的输入数据正则好
- 评估用ddp模拟mp再feature all reduce之前，要不要加一层lora微调，如果要加怎么加

请据此，写出python和/或c++代码。

ファイルを保存してください。

情報科学大学院の履歴書テンプレートのファイルは保存しています。
Markdown形式でファイルを保存していますか？リンクのフォーマットは正しいですか？

このようなテンプレートを生成するのを手伝ってもらえますか？テンプレートの内容を提供してください。また、PDFの形式についても教えてください。フォーマットが少し異なる場合は、履歴書テンプレートのPDFファイルを再作成してください。

在调试一个多智能体系统的提示链时，使用科学的模板是关键。这可以确保每个智能体之间的交互得到充分测试，并且整个系统能够有效地实现目标。以下是一个结构化的提示链模板，可以帮助你科学地调试系统：

1. 环境设置 (Context Setup)
描述环境: 确保每个智能体了解当前的环境，例如场景、任务目标和已知的限制条件。
角色分配: 明确每个智能体的角色和职责。每个智能体应该知道自己的角色以及其他智能体的角色。
2. 初始状态 (Initial State)
输入条件: 定义智能体接收的初始输入，例如传感器数据、外部命令、历史数据等。
目标设定: 设定每个智能体的具体目标。这些目标可以是单一目标，也可以是多个目标。
3. 策略和决策 (Strategy and Decision Making)
策略定义: 每个智能体根据接收到的初始状态信息，制定各自的行动策略。
决策过程: 描述各智能体如何在独立和协作的情况下做出决策。可能涉及的决策过程包括博弈论模型、投票机制、优先级排序等。
4. 交互机制 (Interaction Mechanism)
通信协议: 确定智能体之间如何通信。例如，定义通信格式、频率和带宽限制。
信息共享: 决定什么信息需要共享，什么信息需要保密。对于共享信息，可以使用广播机制或点对点机制。
5. 反馈与学习 (Feedback and Learning)
反馈收集: 每个智能体从环境或其他智能体接收反馈信息，并且记录这些信息用于后续分析。
自适应调整: 智能体根据反馈信息调整其策略。可以考虑引入强化学习、迁移学习等机制，使得系统可以动态适应环境变化。
6. 执行和评估 (Execution and Evaluation)
执行阶段: 执行各智能体的决策，并在环境中观察结果。
评估标准: 使用预定的标准来评估每个智能体及整个系统的表现。标准可以包括任务完成度、资源利用率、通信效率等。
7. 日志记录和分析 (Logging and Analysis)
日志记录: 每个智能体的行为、决策过程和通信内容都应该详细记录。
分析与优化: 对记录的日志进行分析，找出潜在的问题，并优化智能体的策略或交互机制。
8. 实验与验证 (Experimentation and Validation)
实验设计: 设计实验来验证系统的鲁棒性。例如，改变环境条件、增加或减少智能体数量、引入不确定性因素等。
验证测试: 进行多轮测试，验证系统在不同场景下的表现。
9. 迭代优化 (Iterative Optimization)
反馈循环: 根据测试结果调整提示链，优化系统的各个部分。
持续改进: 反复进行测试与优化，直到系统达到预期的表现标准。
通过使用上述模板，你可以科学地调试多智能体系统的提示链，确保每个智能体在整个系统中能够有效协作，并实现预定的目标。

请帮我整理成一个完整的提示链，要求主次分点能够一一对应



服务器/计算节点：
CPU：至少 16 核心的处理器，Intel Xeon Gold 6248R，24核心48线程，基础频率 3.0GHz。或 AMD EPYC，以便同时处理多个任务。
GPU：至少 2 块 NVIDIA Tesla V100 或者 2块 NVIDIA A100 40GB，适合深度学习训练和推理。
RAM：256GB DDR4 ECC RAM，确保处理大规模数据集和复杂模型的能力，更高更好，以支持大型数据集和模型。
存储：4TB NVMe SSD + 12TB 7200RPM SATA，至少 2TB SSD + 10TB HDD，用于快速数据访问和操作系统运行以及数据长期存储和备份。

网络设备：
高速以太网交换机，至少支持 10Gbps 连接，例如Cisco Catalyst 9500，以确保数据快速传输。
如果涉及分布式训练，确保高效的节点间数据交换，以及网络带宽和延迟能满足多节点间通信的需求。

边缘设备（如果智能体需要与真实世界交互）：
根据需要部署一定数量的边缘计算设备，如 Raspberry Pi 4 或 NVIDIA Jetson AGX Xavier，适用于实时数据处理和控制。
根据智能体的感知和行动需求选择：
传感器：
摄像头：Sony IMX219 8MP
温度传感器：DS18B20
距离传感器：HC-SR04 Ultrasonic Sensor
执行器：
伺服电机：MG996R
LED灯：RGB LED Strip Light

辅助设备：
不间断电源供应（UPS），例如APC Smart-UPS 1500VA，以确保关键硬件在电源中断时仍能正常运行。
服务器机柜，例如42U Standard Server Rack Cabinet，用于安装和保护服务器及网络设备。
冷却系统，例如Custom Liquid Cooling Systems，确保硬件设备运行在合适的温度下。

软件与开发工具：
操作系统：选择稳定的服务器操作系统，如 Ubuntu Server 20.04 LTS 或 CentOS。
开发和训练工具：确保所需的深度学习框架（如 TensorFlow 2.x, PyTorch 1.x）和编程语言环境（如 Python）已安装配置。

安全措施：
网络安全：使用Fortinet FortiGate Firewall设置防火墙和网络隔离，尤其是当系统与外部网络连接时。
物理安全：使用电子访问控制系统，保护关键硬件不受未授权访问。

成本估算
服务器/计算节点：约 $26,100 USD
网络设备：约 $7,000 USD
边缘设备：约 $1,050 USD
辅助和安全设备：约 $3,700 USD
总估算成本：约 $37,850 USD

我现在要用这个硬件的配置前去训练多智能体的系统 可以怎么做

# Step 1: Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 2: Load the Shusheng Puyu 7B model and tokenizer
model_name = "shusheng-puyu-7b"  # Replace with actual model path or identifier

# Use the bf16 data type if available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load the tokenizer and model with appropriate precision
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True  # Use this flag for reducing memory usage
).to(device)

# Step 3: Define a function to perform quantized inference
def generate_response(input_text, max_length=100):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate response using model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # Enable sampling for varied responses
            top_p=0.9,  # Top-p sampling
            temperature=0.7  # Temperature setting to control randomness
        )
    
    # Decode and return the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 4: Test the inference with a sample input
if __name__ == "__main__":
    sample_input = "你好，请问你有什么推荐的书籍吗？"
    response = generate_response(sample_input)
    print(f"Model response: {response}")



---
domain: nlp #领域更改为nlp
tags: #自定义标签
- text-classification
- sentiment-analysis
datasets: #关联数据集
  evaluation:
  - your_dataset/evaluation_set
  test:
  - your_dataset/test_set
  train:
  - your_dataset/train_set
models: #关联模型
- your_model/nlp_model

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: MIT License # 修改了许可证为MIT
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/YourName/YourRepository.git
```

### 示例2：更改关联模型和数据集

假设你要更改模型为`iic/ofa_text-generation`，并更改数据集：

```yaml
---
domain: multi-modal # 领域设为多模态
tags: #自定义标签
- multimodal-learning
datasets: #关联数据集
  evaluation:
  - iic/multimodal_evaluation_set
  test:
  - iic/multimodal_test_set
  train:
  - iic/multimodal_train_set
models: #关联模型
- iic/ofa_text-generation

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: main.py # 修改启动文件
license: Apache License 2.0
---
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/YourName/YourNewProject.git
```


### 示例3：添加新的启动文件

如果你要指定一个新的启动文件：

```yaml
---
domain: nlp
tags: #自定义标签
- chatbot
datasets: #关联数据集
  evaluation:
  - my_dataset/evaluation_set
  test:
  - my_dataset/test_set
  train:
  - my_dataset/train_set
models: #关联模型
- my_model/chatbot_model

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
deployspec: # 添加了新的启动文件
  entry_file: run.py
license: GPL-3.0 # 修改许可证为GPL-3.0
---
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/MyName/MyChatbot.git
```
