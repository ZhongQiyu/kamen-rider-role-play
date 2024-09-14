# NVIDIA AI Tutorial and Practical Tasks

本 `README.md` 文件提供了 NVIDIA AI 训练营和在线研讨会的详细信息，包括每日活动安排、学习目标、硬件和软件需求，以及技术难点分析。

## 目录

- [活动简介](#活动简介)
- [日程安排](#日程安排)
- [硬件和软件需求](#硬件和软件需求)
- [技术难点与解决方案](#技术难点与解决方案)
- [学习资源与参考链接](#学习资源与参考链接)

## 活动简介

本训练营旨在通过在线理论学习和实践操作相结合的方式，帮助参与者深入理解和掌握 NVIDIA 的 AI 开发工具和平台，探索 AI 技术的前沿应用。通过参与活动，学员将有机会学习 LLM（大语言模型）、RAG（检索增强生成）、多模态 AI-Agent、Phi3 等多个模块的开发和应用。

## 日程安排

每日上午 10:00~11:00 理论讲解，下午 2:00~4:00 实验和答疑。活动详细安排如下：

| 日期       | 项目内容                                 | 目标                                         |
|------------|-------------------------------------------|----------------------------------------------|
| 2月26日    | **Chat With RTX**                          | CUDA 环境配置，安装 Chat With RTX 应用，体验 LLM 对话 / 编程助手 |
| 2月27日    | **Stable Diffusion XL**                    | 安装 NVIDIA Workbench 应用，在 Jupyter 中调用 SDXL 模型生成图像内容 |
| 2月28日    | **SDXL + LoRA**                            | 在本地训练 LoRA，结合 SDXL 和 LoRA 生成自己专属头像          |
| 2月29日    | **Mistral 编程助手**                        | 在 NVIDIA Workbench 上部署和调试 Mistral 大模型，体验专属编程助手 |
| 3月1日     | **AIPC 应用扩展**                           | 本地安装 NVIDIA Canvas、Omniverse 应用，体验 AIPC 应用      |

## 硬件和软件需求

- **硬件要求**：带有 12G 以上显存的 NVIDIA 30 系或以上显卡的个人 PC。
- **软件要求**：Windows 10 或 Ubuntu 系统，CUDA，NVIDIA Workbench，NVIDIA Chat with RTX。
- **预备技能**：无需任何专业技能，能够独立操作电脑安装软件即可。

## 技术难点与解决方案

### 1. LLM-RAG 工作流程与原理

- **难点**：如何高效实现大语言模型与检索增强生成（RAG）的结合。
- **解决方案**：通过 LangChain 与 NIM 平台的结合，可以实现 LLM-RAG 检索应用。提供代码演示和实际操作练习。

### 2. 多模态模型的开发与应用

- **难点**：如何在边缘设备上高效运行多模态模型，并在 Jetson 平台上实现应用。
- **解决方案**：使用 Phi-3 和 NIM 的结合，优化多模态模型的推理效率，并基于 Gradio 框架创建用户友好的前端互动界面。

### 3. Phi-3 + NIM 平台使用介绍

- **难点**：如何在 Jetson 平台上使用 Phi-3 Vision 构建多模态应用。
- **解决方案**：结合 NVIDIA 推理平台，提升视觉内容生成的效率，代码实战演示 Phi-3-Vision 的推理应用。

### 4. Whisper 与 NIM 构建 RAG 语音平台

- **难点**：如何将 Speech AI 与 LLM-RAG 有效结合，构建语音交互智能体。
- **解决方案**：使用 Whisper 语音识别模型与 Phi3 小模型在 NIM 平台中结合，实现语音平台的高效搭建。

## 学习资源与参考链接

- [NVIDIA Chat with RTX 安装教程](https://zhuanlan.zhihu.com/p/683494847)
- [NVIDIA Workbench 安装教程](https://blog.csdn.net/kunhe0512/article/details/136283665)
- [NVIDIA AI Foundation 构建 LLM-RAG 有声检索智能体](https://live.csdn.net/room/csdnnews/bnm2tfJZ)
- [NVIDIA 官网](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/)
- [TRT-LLM-RAG Windows GitHub](https://github.com/NVIDIA/trt-llm-rag-windows)
- [TRT-LLM-RAG Linux GitHub](https://github.com/noahc1510/trt-llm-rag-linux)

## 结论

通过本次训练营和研讨会，参与者将能掌握 AI 技术的最新进展，并通过实际操作提升自己的开发技能。加入我们，与全球的技术爱好者和专家一起，探索 AI 的未来前沿，释放您的创造潜力！

