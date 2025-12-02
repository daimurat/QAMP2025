# QiskitHumanEval評価スクリプト（RAG統合版）

## 概要

`evaluate_qiskit_humaneval_rag.py`は、CLAPPのRAGバックエンドを統合したQiskitHumanEval自動評価スクリプトです。元の`evaluate_qiskit_humaneval.py`と同じ形式で結果を出力しますが、Qiskitドキュメントのベクトルストアを活用してより正確なコード生成を実現します。

## 主な特徴

- **RAG統合**: QiskitドキュメントのFAISSベクトルストアから関連コンテキストを取得
- **シンプルな評価**: エラー修正ループなしで一貫した評価
- **互換性**: 元の評価スクリプトと同じ出力形式（CSV + JSON）
- **柔軟性**: RAGの有効/無効を切り替え可能

## 前提条件

### 1. 環境セットアップ

```bash
# conda環境をアクティベート
conda activate streamapp

# または、必要なパッケージをインストール
pip install datasets openai langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

### 2. APIキーの設定

`.env`ファイルにOpenAI APIキーを設定:

```bash
OPENAI_API_KEY=your-api-key-here
```

### 3. ベクトルストアの準備

RAGを使用する場合、事前にベクトルストアを生成する必要があります:

```bash
# CLAPPアプリでベクトルストアを生成
streamlit run CLAPP.py
# サイドバーで "Generate Embedding" をクリック

# または、embedding.pyを直接実行
python embedding.py
```

デフォルトでは`faiss_index/`ディレクトリにベクトルストアが保存されます。

## 使用方法

### 基本的な使用例

```bash
# RAGありで評価（デフォルト）
python evaluate_qiskit_humaneval_rag.py

# RAGなしで評価
python evaluate_qiskit_humaneval_rag.py --no-rag

# 特定のモデルで評価
python evaluate_qiskit_humaneval_rag.py --model gpt-4o

# 最初の10タスクのみ評価（テスト用）
python evaluate_qiskit_humaneval_rag.py --max-items 10
```

### 詳細なオプション

```bash
python evaluate_qiskit_humaneval_rag.py \
  --model gpt-4o-mini \
  --dataset Qiskit/qiskit_humaneval \
  --split test \
  --max-items 151 \
  --temperature 0.2 \
  --max-output-tokens 800 \
  --timeout-sec 45 \
  --use-rag \
  --rag-top-k 5 \
  --vector-store-path faiss_index \
  --outdir out
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--model` | `gpt-4o-mini` | OpenAIモデル名 |
| `--dataset` | `Qiskit/qiskit_humaneval` | データセット名 |
| `--split` | `test` | データセット分割 |
| `--max-items` | `None` | 評価するタスク数の制限 |
| `--temperature` | `0.2` | サンプリング温度 |
| `--max-output-tokens` | `800` | 最大出力トークン数 |
| `--timeout-sec` | `45` | テスト実行タイムアウト（秒） |
| `--use-rag` | `True` | RAGを使用 |
| `--no-rag` | - | RAGを無効化 |
| `--rag-top-k` | `5` | RAG検索の上位k件 |
| `--vector-store-path` | `faiss_index` | ベクトルストアのパス |
| `--outdir` | `out` | 出力ディレクトリ |
| `--dry-run` | `False` | モデル呼び出しをスキップ |

## 出力形式

### ディレクトリ構造

```
out/
└── qiskit_humaneval_20251202_143022_gpt-4o-mini_rag/
    ├── generations/
    │   ├── 000_function_name.py
    │   ├── 001_function_name.py
    │   └── ...
    ├── results.csv
    └── summary.json
```

### results.csv

各タスクの詳細な結果:

```csv
task_id,entry_point,passed,error,gen_tokens,prompt_chars,completion_chars,latency_s,difficulty_scale,model,file_path,rag_used
qiskit/0,create_bell_state,True,,245,523,187,1.23,easy,gpt-4o-mini,out/.../000_create_bell_state.py,True
...
```

### summary.json

評価のサマリー:

```json
{
  "model": "gpt-4o-mini",
  "dataset": "Qiskit/qiskit_humaneval",
  "split": "test",
  "timestamp": "20251202_143022",
  "rag_enabled": true,
  "rag_top_k": 5,
  "temperature": 0.2,
  "pass_at_1": 0.7549,
  "passed": 114,
  "total": 151,
  "by_difficulty": {
    "easy": {"passed": 45, "total": 50},
    "medium": {"passed": 42, "total": 60},
    "hard": {"passed": 27, "total": 41}
  }
}
```

## 評価フロー

1. **初期化**
   - OpenAI APIクライアントの初期化
   - ベクトルストアの読み込み（RAG使用時）
   - データセットの読み込み

2. **各タスクの評価**
   - RAGでコンテキスト取得（有効時）
   - LLMでコード生成（1回）
   - テスト実行（1回）
   - 結果記録

3. **結果集計**
   - pass@1の計算
   - 難易度別の集計
   - CSV/JSON出力

## RAGの効果

RAGを使用することで、以下の改善が期待されます:

- **精度向上**: Qiskitドキュメントから関連情報を取得し、より正確な実装を生成
- **API理解**: 最新のQiskit APIの使用方法を参照
- **ベストプラクティス**: ドキュメントに記載されたベストプラクティスを反映

### RAGあり vs RAGなしの比較

```bash
# RAGありで評価
python evaluate_qiskit_humaneval_rag.py --use-rag --max-items 10

# RAGなしで評価
python evaluate_qiskit_humaneval_rag.py --no-rag --max-items 10

# 結果を比較
# out/qiskit_humaneval_*_rag/summary.json
# out/qiskit_humaneval_*_norag/summary.json
```

## トラブルシューティング

### ベクトルストアが見つからない

```
⚠️  Failed to load vector store: [Errno 2] No such file or directory: 'faiss_index'
```

**解決方法**: ベクトルストアを生成してください:

```bash
# CLAPPアプリで生成
streamlit run CLAPP.py

# または、embedding.pyを実行
python embedding.py
```

### APIキーエラー

```
openai.AuthenticationError: Incorrect API key provided
```

**解決方法**: `.env`ファイルにAPIキーを設定してください:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### タイムアウトエラー

一部のタスクでタイムアウトが発生する場合:

```bash
# タイムアウトを延長
python evaluate_qiskit_humaneval_rag.py --timeout-sec 60
```

## 元のスクリプトとの違い

| 機能 | 元のスクリプト | RAG統合版 |
|------|--------------|----------|
| RAG | ❌ | ✅ |
| エラー修正ループ | ❌ | ❌ |
| ベクトルストア | 不要 | 必要（RAG使用時） |
| 依存関係 | OpenAI, datasets | + LangChain, FAISS |
| 出力形式 | CSV + JSON | CSV + JSON（互換） |

## 参考資料

- [QiskitHumanEvalデータセット](https://huggingface.co/datasets/Qiskit/qiskit_humaneval)
- [論文: Qiskit HumanEval](https://arxiv.org/abs/2406.14712)
- [CLAPP: Qiskit LLM Agent](../README.md)
- [元の評価スクリプト](../../01_qiskit-human-eval/evaluate_qiskit_humaneval.py)

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)を参照してください。