
# Multi-Agent System for Kamen Rider Blade

Welcome to the Multi-Agent System project for Kamen Rider Blade. This README provides guidance in English, Japanese, and Chinese.

**Choose Your Language:**
- [English](#english)
- [日本語 (Japanese)](#japanese)
- [中文 (Chinese)](#chinese)

## English

This project utilizes question-answering with multi-agent systems dedicated to "Kamen Rider Blade". It ensures originality and no reason to abstain from engagement—your presence is the "Last Trump Card."

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/stage-play-llm.git
   ```
2. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data by placing related stage play data in `data/raw/` and run the preprocessing script:
   ```bash
   python scripts/preprocessing.py
   ```
4. Train the model:
   ```bash
   python scripts/train.py
   ```
5. Fine-tune the model:
   ```bash
   python scripts/finetune.py
   ```
6. Evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```

---

## 日本語 (Japanese)


# 仮面ライダー剣: マルチエージェントシステム

## プロジェクト概要

これは、「仮面ライダー剣」に特化した質問応答を利用した再現可能なプロジェクトです。独自性が保証され、参加を控える理由はありません。あなたの存在が「最後の切り札」です。

**更新通知:** 7月19日

## 始め方

### 環境設定

1. リポジトリをクローンする:
   ```bash
   git clone https://github.com/your-repository/stage-play-llm.git
   ```
2. 必要な依存関係をインストールする:
   ```bash
   pip install -r requirements.txt
   ```
3. データ準備: 関連するステージプレイデータを `data/raw/` に配置し、前処理スクリプトを実行する:
   ```bash
   python scripts/preprocessing.py
   ```
4. モデルトレーニング:
   ```bash
   python scripts/train.py
   ```
5. モデルファインチューニング:
   ```bash
   python scripts/finetune.py
   ```
6. モデル評価:
   ```bash
   python scripts/evaluate.py
   ```

## 技術スタック

- TensorFlowまたはPyTorch
- LangChain
- Python 3.x

## コントリビューターガイド

プロジェクトへの改善提案やコードの貢献については、`CONTRIBUTING.md`を参照してください。

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は `LICENSE` ファイルを参照してください。

## 連絡先

質問がある場合やサポートが必要な場合は、xiaoyu991214@gmail.com までご連絡ください。

---

## モジュール

### エージェント通信 (agent_comm)

エージェント間の非同期通信を容易にし、マルチエージェント協調をサポートします。

### データプロセッサ (data_processor)

入力データを分析用に処理し、マルチエージェントシステム用の非同期データ処理をサポートします。

---

## APIドキュメント

### データプロセッサ API

#### GET
- **説明**: 処理済みデータを取得します。
- **パラメータ**:
  - `data_id` (str): データの一意の識別子です。
- **レスポンス**:
  - `processed_data` (JSON): 処理済みデータのJSON形式です。

#### POST
- **説明**: データの処理を依頼します。
- **パラメータ**:
  - `raw_data` (JSON): 処理する生データです。
- **レスポンス**:
  - `processing_id` (str): 処理ジョブの一意の識別子です。

#### POST (非同期)
- **説明**: 非同期でデータの処理を依頼します。
- **パラメータ**:
  - `raw_data` (JSON): 処理する生データです。
  - `callback_url` (str): 非同期結果のコールバックURLです。
- **レスポンス**:
  - `processing_id` (str): 非同期処理ジョブの一意の識別子です。

### エージェント通信 API

#### GET
- **説明**: 特定のエージェントのメッセージを取得します。
- **パラメータ**:
  - `agent_id` (str): エージェントの識別子です。
- **レスポンス**:
  - `messages` (JSON): エージェントが受信したメッセージのリストです。

#### POST
- **説明**: エージェント間の通信を開始します。
- **パラメータ**:
  - `agent_id` (str): 通信を開始するエージェントの識別子です。
  - `message` (str): 送信されるメッセージです。
- **レスポンス**:
  - `confirmation_message` (str): 通信が開始されたことを確認するメッセージです。

#### POST (非同期)
- **説明**: エージェント間の非同期通信を開始します。
- **パラメータ**:
  - `agent_id` (str): 通信を開始するエージェントの識別子です。
  - `message` (str): 送信されるメッセージです。
  - `callback_url` (str): 非同期応答のコールバックURLです。
- **レスポンス**:
  - `confirmation_message` (str): 非同期通信が開始されたことを確認するメッセージです。


---

## 中文 (Chinese)


# 假面骑士剑：多智能体系统

## 项目概览

这是一个利用问答的可复制项目，专用于“假面骑士剑”。它保证了原创性，没有理由回避参与——你的存在是“最后的王牌”。

**更新通知：**7月19日

## 入门

### 环境设置

1. 克隆仓库：
   ```bash
   git clone https://github.com/your-repository/stage-play-llm.git
   ```
2. 安装必要的依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 数据准备：将相关的舞台剧数据放置在`data/raw/`目录下，并运行预处理脚本：
   ```bash
   python scripts/preprocessing.py
   ```
4. 模型训练：
   ```bash
   python scripts/train.py
   ```
5. 模型微调：
   ```bash
   python scripts/finetune.py
   ```
6. 模型评估：
   ```bash
   python scripts/evaluate.py
   ```

## 技术栈

- TensorFlow或PyTorch
- LangChain
- Python 3.x

## 贡献者指南

我们欢迎更多的开发者加入我们的项目。如果您对改进此项目有任何建议或想要贡献代码，请阅读`CONTRIBUTING.md`。

## 许可证

此项目采用 MIT 许可证。详情请见`LICENSE`文件。

## 联系方式

如有问题或需要支持，请联系xiaoyu991214@gmail.com。

---

## 模块

### 代理通信 (agent_comm)

促进代理之间的异步通信，并支持多代理协作。

### 数据处理器 (data_processor)

处理输入数据以供分析，并支持多代理系统的异步数据处理。

---

## API文档

### 数据处理器 API

#### GET
- **描述**：检索处理后的数据。
- **参数**：
  - `data_id` (str)：数据的唯一标识符。
- **响应**：
  - `processed_data` (JSON)：处理后的数据的JSON格式。

#### POST
- **描述**：提交数据以供处理。
- **参数**：
  - `raw_data` (JSON)：待处理的原始数据。
- **响应**：
  - `processing_id` (str)：处理任务的唯一标识符。

#### POST (异步)
- **描述**：提交数据以供异步处理。
- **参数**：
  - `raw_data` (JSON)：待处理的原始数据。
  - `callback_url` (str)：异步结果的回调URL。
- **响应**：
  - `processing_id` (str)：异步处理任务的唯一标识符。

### 代理通信 API

#### GET
- **描述**：检索指定代理的消息。
- **参数**：
  - `agent_id` (str)：代理的标识符。
- **响应**：
  - `messages` (JSON)：代理收到的消息列表。

#### POST
- **描述**：启动代理之间的通信。
- **参数**：
  - `agent_id` (str)：启动通信的代理的标识符。
  - `message` (str)：要发送的消息。
- **响应**：
  - `confirmation_message` (str)：确认已启动通信的消息。

#### POST (异步)
- **描述**：启动代理之间的异步通信。
- **参数**：
  - `agent_id` (str)：启动通信的代理的标识符。
  - `message` (str)：要发送的消息。
  - `callback_url` (str)：异步响应的回调URL。
- **响应**：
  - `confirmation_message` (str)：确认已启动异步通信的消息。


