# 仮面ライダー剣マルチエージェントシステム使用ガイド

## システム概要
このマルチエージェントシステムは、日本の特撮ドラマ「仮面ライダー剣」の主要キャラクターに基づいて設計されており、多角的な問題解決と意思決定支援ツールを提供することを目的としています。システムには4つのコアエージェントが含まれており、各エージェントはドラマのキャラクターの特徴と能力をシミュレートし、ユーザーが複雑な問題や状況を処理するのを支援します。

## エージェント紹介

### 剣崎一真エージェント
- 主な機能：意思決定と行動実行
- 特徴：勇敢、正義感が強い、適応力が高い
- 使用場面：迅速な決定と果断な行動が必要な状況

### 橘朔也エージェント
- 主な機能：戦略分析とリスク評価
- 特徴：冷静、理性的、慎重
- 使用場面：詳細な分析と長期的な計画が必要な状況

### 相川始エージェント
- 主な機能：情報収集と情報処理
- 特徴：鋭敏、観察力が高い、柔軟
- 使用場面：広範な情報収集と詳細な洞察が必要な状況

### 上城睦月エージェント
- 主な機能：支援調整と感情分析
- 特徴：温和、同情心が豊か、洞察力が高い
- 使用場面：対人関係や感情的要因を扱う状況

## システムの使用方法

### エージェント通信 (agent_comm)
エージェント間の非同期通信を容易にし、マルチエージェント協調をサポートします。

### データプロセッサ (data_processor)
入力データを分析用に処理し、マルチエージェントシステム用の非同期データ処理をサポートします。

## データプロセッサ API

### GET
- **説明**: 処理済みデータを取得します。
- **パラメータ**: `data_id` (str): データの一意の識別子です。
- **レスポンス**: `processed_data` (JSON): 処理済みデータのJSON形式です。

### POST
- **説明**: データの処理を依頼します。
- **パラメータ**: `raw_data` (JSON): 処理する生データです。
- **レスポンス**: `processing_id` (str): 処理ジョブの一意の識別子です。

### POST (非同期)
- **説明**: 非同期でデータの処理を依頼します。
- **パラメータ**: `raw_data` (JSON), `callback_url` (str)
- **レスポンス**: `processing_id` (str)

### エージェント通信 API

### GET
- **説明**: 特定のエージェントのメッセージを取得します。
- **パラメータ**: `agent_id` (str)
- **レスポンス**: `messages` (JSON)

### POST
- **説明**: エージェント間の通信を開始します。
- **パラメータ**: `agent_id` (str), `message` (str)
- **レスポンス**: `confirmation_message` (str)

### POST (非同期)
- **説明**: エージェント間の非同期通信を開始します。
- **パラメータ**: `agent_id` (str), `message` (str), `callback_url` (str)
- **レスポンス**: `confirmation_message` (str)

## システムの初期化
```python
from blade_agents import BladeAgentSystem

system = BladeAgentSystem()
system.initialize()
```

## タスク設定とエージェント活性化
```python
# 問題の設定
system.set_task("コミュニケーション効率をどのように改善すべきか？")

# 剣崎エージェントの活性化と決定の取得
decision = system.activate_agent("kazuki")
print("剣崎の決定:", decision)
```

## 総合分析
```python
final_solution = system.synthesize_solutions()
print("最終解決策:", final_solution)
```

## カスタマイズと拡張
```python
# エージェントパラメータの調整
system.customize_agent("kazuki", risk_tolerance=0.8)

# 新機能の追加
system.add_new_capability("aikawa", "social_media_analysis")
```
