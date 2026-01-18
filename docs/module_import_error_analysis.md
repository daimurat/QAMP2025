# ModuleNotFoundError 分析レポート

## エラー内容

```
Traceback (most recent call last):
  File "/Users/daimurat/workspace/quantum/QAMP2025/scripts/evaluate_qiskit_humaneval.py", line 19, in <module>
    from src.application.workflows.deep_thought import run_deep_thought_mode
ModuleNotFoundError: No module named 'src'
```

## 根本原因

### 1. Python パス問題
スクリプトを `python scripts/evaluate_qiskit_humaneval.py` で実行すると、Pythonインタープリターは**実行したディレクトリ**（プロジェクトルート `/Users/daimurat/workspace/quantum/QAMP2025`）を `sys.path` に追加します。

しかし、[`evaluate_qiskit_humaneval.py`](scripts/evaluate_qiskit_humaneval.py:19-20) は以下のように絶対インポートを使用しています：

```python
from src.application.workflows.deep_thought import run_deep_thought_mode
from src.application.workflows.fast import run_fast_mode
```

### 2. プロジェクト構造の確認
プロジェクト構造：
```
/Users/daimurat/workspace/quantum/QAMP2025/
├── src/
│   ├── application/
│   │   └── workflows/
│   │       ├── deep_thought.py
│   │       └── fast.py
│   └── (no __init__.py)
└── scripts/
    └── evaluate_qiskit_humaneval.py
```

**重要な発見**: `src/` ディレクトリに `__init__.py` ファイルが存在しません。

### 3. 他のファイルとの比較
[`app.py`](app.py:12-16) は同じインポート方法で正常に動作しています：

```python
from src.utils.session_state import SessionStateManager
from src.presentation.components import inject_global_styles_and_font, render_header
```

これは `app.py` がプロジェクトルートから実行されるため、`src` モジュールが見つかるからです。

## 解決策

### 解決策 1: `__init__.py` ファイルを追加（推奨）
`src/` ディレクトリを正式なPythonパッケージにする：

**必要なファイル**:
- `src/__init__.py`
- `src/application/__init__.py`
- `src/application/workflows/__init__.py`

**メリット**:
- プロジェクト全体で一貫したインポート方法
- Pythonのベストプラクティスに準拠
- 他のツール（pytest、mypy等）との互換性向上

**デメリット**:
- 複数のファイルを作成する必要がある

### 解決策 2: スクリプト内でパスを動的に追加
[`evaluate_qiskit_humaneval.py`](scripts/evaluate_qiskit_humaneval.py:1-8) の先頭に以下を追加：

```python
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

**メリット**:
- 最小限の変更で即座に解決
- スクリプト単体で完結

**デメリット**:
- 各スクリプトに同じコードを追加する必要がある
- 保守性が低い

### 解決策 3: Python モジュールとして実行
実行方法を変更：

```bash
# プロジェクトルートから
python -m scripts.evaluate_qiskit_humaneval
```

**メリット**:
- コード変更不要
- Pythonの標準的な実行方法

**デメリット**:
- `scripts/__init__.py` が必要
- 実行コマンドが長くなる

### 解決策 4: 相対インポートに変更
インポート文を相対パスに変更（非推奨）：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.application.workflows.deep_thought import run_deep_thought_mode
```

**デメリット**:
- プロジェクト構造に依存
- 保守性が低い

## 推奨アクション

**最も推奨される解決策**: **解決策 1 + 解決策 2 の組み合わせ**

1. `src/` とそのサブディレクトリに `__init__.py` を追加してプロジェクト全体を整理
2. スクリプトにもパス追加コードを含めて、どこから実行しても動作するようにする

これにより：
- プロジェクト構造がPythonのベストプラクティスに準拠
- スクリプトの実行場所に依存しない堅牢性
- 将来的な拡張性の確保

## 次のステップ

1. 必要な `__init__.py` ファイルを作成
2. [`evaluate_qiskit_humaneval.py`](scripts/evaluate_qiskit_humaneval.py:1-10) にパス追加コードを挿入
3. スクリプトを再実行してエラーが解消されることを確認