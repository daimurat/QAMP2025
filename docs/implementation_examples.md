# StreamlitベースRAGアプリケーション実装例

このドキュメントでは、推奨ディレクトリ構成に基づいた具体的な実装例を提供します。

## 目次
1. [基本的なRAGアプリケーション](#1-基本的なragアプリケーション)
2. [設定管理の実装](#2-設定管理の実装)
3. [RAGコアロジックの実装](#3-ragコアロジックの実装)
4. [UIコンポーネントの実装](#4-uiコンポーネントの実装)
5. [サービス層の実装](#5-サービス層の実装)
6. [完全な実装例](#6-完全な実装例)

---

## 1. 基本的なRAGアプリケーション

### メインエントリーポイント (`app.py`)

```python
"""
StreamlitベースRAGアプリケーションのメインエントリーポイント
"""
import streamlit as st
from dotenv import load_dotenv

from config.settings import Settings
from src.presentation.components.chat_interface import render_chat_interface
from src.presentation.components.sidebar import render_sidebar
from src.utils.session_state import SessionStateManager

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """メイン関数"""
    # 設定の初期化
    settings = Settings()
    
    # セッション状態の初期化
    SessionStateManager.initialize()
    
    # ヘッダー
    st.title("🤖 RAG Chat Assistant")
    st.markdown("---")
    
    # サイドバー
    render_sidebar()
    
    # チャットインターフェース
    render_chat_interface()

if __name__ == "__main__":
    main()
```

---

## 2. 設定管理の実装

### 設定クラス (`config/settings.py`)

```python
"""
アプリケーション設定管理
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    # LLM Settings
    llm_provider: str = Field("openai", env="LLM_PROVIDER")
    llm_model: str = Field("gpt-4o-mini", env="LLM_MODEL")
    temperature: float = Field(0.7, env="TEMPERATURE")
    max_tokens: int = Field(2000, env="MAX_TOKENS")
    
    # Embedding Settings
    embedding_provider: str = Field("openai", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # RAG Settings
    top_k: int = Field(5, env="RAG_TOP_K")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    
    # Database Settings
    vector_db_path: str = Field("data/vector_db", env="VECTOR_DB_PATH")
    vector_db_type: str = Field("faiss", env="VECTOR_DB_TYPE")  # faiss, chroma, pinecone
    
    # Cache Settings
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # seconds
    
    # Application Settings
    app_name: str = Field("RAG Chat Assistant", env="APP_NAME")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# シングルトンインスタンス
settings = Settings()
```

### 定数定義 (`config/constants.py`)

```python
"""
アプリケーション定数
"""

# サポートされるLLMプロバイダー
LLM_PROVIDERS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet"]
}

# サポートされる埋め込みモデル
EMBEDDING_MODELS = {
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "gemini": ["gemini-embedding-001"],
    "cohere": ["embed-english-v3.0", "embed-multilingual-v3.0"]
}

# ベクトルDBタイプ
VECTOR_DB_TYPES = ["faiss", "chroma", "pinecone", "qdrant"]

# チャットロール
CHAT_ROLES = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "SYSTEM": "system"
}

# セッションステートキー
SESSION_KEYS = {
    "MESSAGES": "messages",
    "CHAT_HISTORY": "chat_history",
    "RETRIEVER": "retriever",
    "LLM_CLIENT": "llm_client",
    "SETTINGS": "settings"
}
```

### プロンプトテンプレート (`config/prompts/rag_prompts.py`)

```python
"""
RAGシステム用プロンプトテンプレート
"""

SYSTEM_PROMPT = """あなたは親切で知識豊富なAIアシスタントです。
提供されたコンテキスト情報を使用して、ユーザーの質問に正確に答えてください。

重要なルール:
1. コンテキストに基づいて回答してください
2. コンテキストに情報がない場合は、正直にそう伝えてください
3. 推測や創作をしないでください
4. 回答は明確で簡潔にしてください
"""

RAG_QUERY_TEMPLATE = """以下のコンテキスト情報を使用して質問に答えてください。

コンテキスト:
{context}

質問: {query}

回答:"""

NO_CONTEXT_RESPONSE = """申し訳ございませんが、提供された情報の中に質問に答えるための関連情報が見つかりませんでした。
別の質問をしていただくか、より具体的な質問をしていただけますか？"""

CLARIFICATION_PROMPT = """質問が不明確です。以下の点を明確にしていただけますか？

{clarification_points}"""
```

---

## 3. RAGコアロジックの実装

### ドキュメントモデル (`src/domain/models/document.py`)

```python
"""
ドキュメントドメインモデル
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class Document:
    """ドキュメントモデル"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[list] = None
    score: Optional[float] = None
    
    @property
    def source(self) -> str:
        """ソースを取得"""
        return self.metadata.get("source", "unknown")
    
    @property
    def title(self) -> str:
        """タイトルを取得"""
        return self.metadata.get("title", "Untitled")
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score
        }

@dataclass
class Query:
    """クエリモデル"""
    text: str
    embedding: Optional[list] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

@dataclass
class RAGResponse:
    """RAG応答モデル"""
    query: str
    answer: str
    documents: list[Document]
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
```

### 埋め込み生成 (`src/domain/rag/embedder.py`)

```python
"""
埋め込みベクトル生成
"""
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """埋め込み生成の基底クラス"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """単一テキストの埋め込みを生成"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """複数テキストの埋め込みを生成"""
        pass

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI埋め込み生成"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed_text(self, text: str) -> np.ndarray:
        """単一テキストの埋め込みを生成"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """複数テキストの埋め込みを生成"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [np.array(item.embedding) for item in response.data]

class GeminiEmbedder(BaseEmbedder):
    """Gemini埋め込み生成"""
    
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    def embed_text(self, text: str) -> np.ndarray:
        """単一テキストの埋め込みを生成"""
        response = self.client.models.embed_content(
            model=self.model,
            content=text
        )
        return np.array(response.embeddings[0].values)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """複数テキストの埋め込みを生成"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
```

### RAG検索 (`src/domain/rag/retriever.py`)

```python
"""
RAG検索ロジック
"""
from typing import List, Optional
import numpy as np

from src.domain.models.document import Document, Query
from src.domain.rag.embedder import BaseEmbedder
from src.infrastructure.database.vector_store import VectorStore

class RAGRetriever:
    """RAG検索クラス"""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[Document]:
        """クエリに関連するドキュメントを検索"""
        # top_kのデフォルト値
        k = top_k if top_k is not None else self.top_k
        
        # クエリの埋め込みを生成
        query_embedding = self.embedder.embed_text(query)
        
        # ベクトル検索
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filters=filters
        )
        
        # 類似度でフィルタリング
        filtered_results = [
            doc for doc in results
            if doc.score >= self.similarity_threshold
        ]
        
        return filtered_results
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> str:
        """コンテキスト文字列を取得"""
        documents = self.retrieve(query, top_k=top_k)
        
        if not documents:
            return ""
        
        # ドキュメントテキストを結合
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {doc.source}\n"
                f"Content: {doc.text}"
            )
        
        return separator.join(context_parts)
```

### LLM統合 (`src/domain/llm/openai_client.py`)

```python
"""
OpenAI LLMクライアント
"""
from typing import List, Optional, Dict, Any
from openai import OpenAI

class OpenAIClient:
    """OpenAI LLMクライアント"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """テキスト生成"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=stream
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """コンテキスト付きで生成"""
        messages = []
        
        # システムプロンプト
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # ユーザーメッセージ（コンテキスト + クエリ）
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return self.generate(messages)
```

---

## 4. UIコンポーネントの実装

### チャットインターフェース (`src/presentation/components/chat_interface.py`)

```python
"""
チャットインターフェースコンポーネント
"""
import streamlit as st
from typing import Optional

from src.application.services.chat_service import ChatService
from src.utils.session_state import SessionStateManager

def render_chat_interface():
    """チャットインターフェースをレンダリング"""
    
    # チャット履歴の表示
    _display_chat_history()
    
    # 入力フィールド
    if prompt := st.chat_input("質問を入力してください..."):
        _process_user_input(prompt)

def _display_chat_history():
    """チャット履歴を表示"""
    messages = SessionStateManager.get("messages", [])
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # ソース情報の表示（アシスタントメッセージの場合）
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 参照ソース"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['title']}**")
                        st.caption(f"Score: {source['score']:.3f}")
                        st.text(source['text'][:200] + "...")
                        st.markdown("---")

def _process_user_input(user_input: str):
    """ユーザー入力を処理"""
    # ユーザーメッセージを追加
    SessionStateManager.append_message("user", user_input)
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # アシスタント応答を生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            # チャットサービスを使用
            chat_service = ChatService()
            response = chat_service.process_query(user_input)
            
            # 応答を表示
            st.markdown(response["answer"])
            
            # ソース情報を保存
            SessionStateManager.append_message(
                "assistant",
                response["answer"],
                sources=response.get("sources", [])
            )
            
            # ソース情報を表示
            if response.get("sources"):
                with st.expander("📚 参照ソース"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**{i}. {source['title']}**")
                        st.caption(f"Score: {source['score']:.3f}")
                        st.text(source['text'][:200] + "...")
                        st.markdown("---")
```

### サイドバー (`src/presentation/components/sidebar.py`)

```python
"""
サイドバーコンポーネント
"""
import streamlit as st
from config.settings import settings
from config.constants import LLM_PROVIDERS
from src.utils.session_state import SessionStateManager

def render_sidebar():
    """サイドバーをレンダリング"""
    with st.sidebar:
        st.title("⚙️ 設定")
        
        # API Key設定
        _render_api_key_section()
        
        st.markdown("---")
        
        # モデル設定
        _render_model_settings()
        
        st.markdown("---")
        
        # RAG設定
        _render_rag_settings()
        
        st.markdown("---")
        
        # チャットリセット
        if st.button("🗑️ チャット履歴をクリア", use_container_width=True):
            SessionStateManager.clear_messages()
            st.rerun()

def _render_api_key_section():
    """API Key設定セクション"""
    st.subheader("🔑 API Key")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=SessionStateManager.get("api_key", ""),
        help="OpenAI APIキーを入力してください"
    )
    
    if api_key:
        SessionStateManager.set("api_key", api_key)
        st.success("✅ API Key設定済み")

def _render_model_settings():
    """モデル設定セクション"""
    st.subheader("🤖 モデル設定")
    
    # LLMプロバイダー選択
    provider = st.selectbox(
        "LLMプロバイダー",
        options=list(LLM_PROVIDERS.keys()),
        index=0
    )
    
    # モデル選択
    model = st.selectbox(
        "モデル",
        options=LLM_PROVIDERS[provider],
        index=0
    )
    
    # Temperature設定
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="高いほど創造的、低いほど決定的"
    )
    
    # 設定を保存
    SessionStateManager.set("llm_provider", provider)
    SessionStateManager.set("llm_model", model)
    SessionStateManager.set("temperature", temperature)

def _render_rag_settings():
    """RAG設定セクション"""
    st.subheader("📚 RAG設定")
    
    # Top-K設定
    top_k = st.slider(
        "検索ドキュメント数 (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="検索する関連ドキュメントの数"
    )
    
    # 類似度閾値
    similarity_threshold = st.slider(
        "類似度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="この値以上の類似度のドキュメントのみ使用"
    )
    
    # 設定を保存
    SessionStateManager.set("top_k", top_k)
    SessionStateManager.set("similarity_threshold", similarity_threshold)
```

---

## 5. サービス層の実装

### チャットサービス (`src/application/services/chat_service.py`)

```python
"""
チャットサービス
"""
from typing import Dict, Any, List
import streamlit as st

from src.domain.rag.retriever import RAGRetriever
from src.domain.llm.openai_client import OpenAIClient
from src.domain.models.document import Document
from config.prompts.rag_prompts import SYSTEM_PROMPT, RAG_QUERY_TEMPLATE
from src.utils.session_state import SessionStateManager

class ChatService:
    """チャットサービスクラス"""
    
    def __init__(self):
        self.retriever = self._get_retriever()
        self.llm_client = self._get_llm_client()
    
    def _get_retriever(self) -> RAGRetriever:
        """RAG検索エンジンを取得"""
        # セッションから取得または新規作成
        if "retriever" not in st.session_state:
            from src.domain.rag.embedder import OpenAIEmbedder
            from src.infrastructure.database.vector_store import FAISSVectorStore
            from config.settings import settings
            
            embedder = OpenAIEmbedder(
                api_key=SessionStateManager.get("api_key"),
                model=settings.embedding_model
            )
            
            vector_store = FAISSVectorStore(
                index_path=settings.vector_db_path
            )
            
            retriever = RAGRetriever(
                embedder=embedder,
                vector_store=vector_store,
                top_k=SessionStateManager.get("top_k", 5),
                similarity_threshold=SessionStateManager.get("similarity_threshold", 0.7)
            )
            
            st.session_state.retriever = retriever
        
        return st.session_state.retriever
    
    def _get_llm_client(self) -> OpenAIClient:
        """LLMクライアントを取得"""
        if "llm_client" not in st.session_state:
            client = OpenAIClient(
                api_key=SessionStateManager.get("api_key"),
                model=SessionStateManager.get("llm_model", "gpt-4o-mini"),
                temperature=SessionStateManager.get("temperature", 0.7)
            )
            st.session_state.llm_client = client
        
        return st.session_state.llm_client
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """クエリを処理して応答を生成"""
        # 1. 関連ドキュメントを検索
        documents = self.retriever.retrieve(query)
        
        # 2. コンテキストを構築
        context = self._build_context(documents)
        
        # 3. LLMで応答を生成
        answer = self.llm_client.generate_with_context(
            query=query,
            context=context,
            system_prompt=SYSTEM_PROMPT
        )
        
        # 4. 結果を返す
        return {
            "answer": answer,
            "sources": [doc.to_dict() for doc in documents],
            "context": context
        }
    
    def _build_context(self, documents: List[Document]) -> str:
        """ドキュメントからコンテキストを構築"""
        if not documents:
            return "関連する情報が見つかりませんでした。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {doc.source}\n"
                f"Content: {doc.text}"
            )
        
        return "\n\n".join(context_parts)
```

---

## 6. 完全な実装例

### プロジェクト構造

```
rag-chat-app/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── constants.py
│   └── prompts/
│       ├── __init__.py
│       └── rag_prompts.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── presentation/
│   │   ├── __init__.py
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── chat_interface.py
│   │       └── sidebar.py
│   │
│   ├── application/
│   │   ├── __init__.py
│   │   └── services/
│   │       ├── __init__.py
│   │       └── chat_service.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── document.py
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py
│   │   │   └── retriever.py
│   │   └── llm/
│   │       ├── __init__.py
│   │       └── openai_client.py
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   └── database/
│   │       ├── __init__.py
│   │       └── vector_store.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── session_state.py
│
├── data/
│   └── vector_db/
│       └── index.faiss
│
└── scripts/
    └── build_index.py
```

### requirements.txt

```txt
streamlit>=1.30.0
openai>=1.10.0
google-generativeai>=0.3.0
faiss-cpu>=1.7.4
numpy>=1.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-community>=0.0.10
```

### .env.example

```env
# API Keys
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2000

# Embedding Settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# RAG Settings
RAG_TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SIMILARITY_THRESHOLD=0.7

# Database Settings
VECTOR_DB_PATH=data/vector_db
VECTOR_DB_TYPE=faiss

# Application Settings
APP_NAME=RAG Chat Assistant
DEBUG_MODE=false
```

### インデックス構築スクリプト (`scripts/build_index.py`)

```python
"""
ベクトルインデックス構築スクリプト
"""
import os
import sys
from pathlib import Path
import numpy as np
import faiss

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.rag.embedder import OpenAIEmbedder
from config.settings import settings

def load_documents(data_dir: str) -> list:
    """ドキュメントを読み込み"""
    documents = []
    data_path = Path(data_dir)
    
    for file_path in data_path.glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "id": str(file_path),
                "text": text,
                "metadata": {
                    "source": str(file_path),
                    "title": file_path.stem
                }
            })
    
    return documents

def build_faiss_index(documents: list, output_path: str):
    """FAISSインデックスを構築"""
    # 埋め込み生成
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model
    )
    
    print(f"Generating embeddings for {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_texts(texts)
    
    # NumPy配列に変換
    embedding_matrix = np.array(embeddings).astype('float32')
    
    # FAISSインデックスを作成
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    
    # インデックスを保存
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_dir / "index.faiss"))
    
    # メタデータを保存
    import json
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Index built successfully: {output_path}")
    print(f"Total documents: {len(documents)}")
    print(f"Embedding dimension: {dimension}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--output", default="data/vector_db", help="Output directory")
    
    args = parser.parse_args()
    
    documents = load_documents(args.data_dir)
    build_faiss_index(documents, args.output)
```

### 使用方法

1. **環境設定**
```bash
# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# 環境変数を設定
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

2. **インデックス構築**
```bash
# ドキュメントからベクトルインデックスを構築
python scripts/build_index.py --data-dir /path/to/documents --output data/vector_db
```

3. **アプリケーション起動**
```bash
streamlit run app.py
```

---

## まとめ

この実装例では以下を提供しました：

1. **設定管理**: Pydanticを使用した型安全な設定管理
2. **レイヤー分離**: Presentation、Application、Domain、Infrastructureの明確な分離
3. **RAGコア**: 埋め込み生成、ベクトル検索、LLM統合の実装
4. **UIコンポーネント**: 再利用可能なStreamlitコンポーネント
5. **サービス層**: ビジネスロジックのカプセル化
6. **完全な例**: すぐに使える完全なプロジェクト構造

この構造により、保守性、拡張性、テスタビリティの高いRAGアプリケーションを構築できます。