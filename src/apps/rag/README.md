下面是一份**极其基础**、但能跑通端到端流程的 RAG 框架代码（本地向量化 → FAISS 检索 → 可选重排 → 生成）。

---

# 目录结构

```
rag-basic/
├── README.md
├── requirements.txt
├── config.py
├── build_index.py          # 构建索引（离线）
├── query.py                # 运行一次查询（在线）
└── rag/
    ├── __init__.py
    ├── chunks.py           # 文本分块
    ├── embedder.py         # 向量化
    ├── index.py            # 索引的保存/加载（FAISS + 元数据）
    ├── retriever.py        # 初检索（top-k）
    ├── reranker.py         # （可选）重排
    ├── generator.py        # （可选）生成（OpenAI 或本地HF二选一）
    └── pipeline.py         # 串起RAG流程
```

> **数据输入**：默认从 `./data/docs/` 读取 `.txt/.md` 文档（示例中按需自备）。

---

# requirements.txt

```txt
faiss-cpu
sentence-transformers
numpy
pydantic
tqdm
transformers
accelerate
# 如需OpenAI推理，取消下一行注释并配置环境变量：
# openai
```

---

# README.md

````md
# 极简 RAG 框架

## 1) 安装依赖
```bash
pip install -r requirements.txt
````

## 2) 准备数据

将你的 `.txt` 或 `.md` 文件放入 `./data/docs/` 目录。

## 3) 构建索引（离线）

```bash
python build_index.py --data_dir ./data/docs --index_dir ./data/index
```

## 4) 运行查询（在线）

```bash
python query.py "什么是RAG？" --index_dir ./data/index --top_k 5 --with_rerank
```

> 生成阶段默认走“模板式总结”。
> 如需调用 OpenAI：设置环境变量 `OPENAI_API_KEY`，并在 `config.py` 里将 `GEN_PROVIDER="openai"`。
> 如需用本地 HF 小模型：`GEN_PROVIDER="hf"`（需首次下载模型）。

````

---

# config.py
```python
from pydantic import BaseModel

class Settings(BaseModel):
    # 嵌入模型（轻量、够用）
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 重排：可选 CrossEncoder（如未安装将自动退化为余弦）
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    # 生成模型提供方："none" | "openai" | "hf"
    GEN_PROVIDER: str = "none"  # 初学可先用 none（模板总结）

    # HuggingFace 文本生成模型（示例）
    HF_GEN_MODEL: str = "google/flan-t5-base"

    # OpenAI（如需）
    OPENAI_MODEL: str = "gpt-4o-mini"

settings = Settings()
````

---

# rag/**init**.py

```python
# 空文件即可
```

---

# rag/chunks.py

```python
from __future__ import annotations
from typing import List

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """按字数切分，含简单重叠窗口。适合入门，后续可替换为按句/段/Markdown结构切分。"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
```

---

# rag/embedder.py

```python
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        embs = self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embs.astype(np.float32)
```

---

# rag/index.py

```python
from __future__ import annotations
import os, json
import numpy as np
import faiss
from typing import List, Dict

META_FILE = "meta.jsonl"     # 每行：{"id": int, "text": str, "source": str}
VEC_FILE = "vectors.npy"     # 按行与 meta 对齐
INDEX_FILE = "faiss.index"   # FAISS 索引

class FaissIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.meta_path = os.path.join(index_dir, META_FILE)
        self.vec_path = os.path.join(index_dir, VEC_FILE)
        self.faiss_path = os.path.join(index_dir, INDEX_FILE)
        self.index = None
        self.dim = None

    def build(self, vectors: np.ndarray):
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # 余弦相似：已在Embedder中归一化
        self.index.add(vectors)
        faiss.write_index(self.index, self.faiss_path)
        np.save(self.vec_path, vectors)

    def load(self):
        self.index = faiss.read_index(self.faiss_path)
        self.dim = self.index.d

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        if self.index is None:
            self.load()
        scores, idxs = self.index.search(query_vec.astype(np.float32), top_k)
        return scores[0], idxs[0]

    def save_meta(self, rows: List[Dict]):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load_meta(self):
        metas = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                metas.append(json.loads(line))
        return metas
```

---

# rag/retriever.py

```python
from __future__ import annotations
from typing import List, Dict
import numpy as np

from .embedder import Embedder
from .index import FaissIndex

class Retriever:
    def __init__(self, index_dir: str, embed_model: str):
        self.embedder = Embedder(embed_model)
        self.index = FaissIndex(index_dir)
        self.index.load()
        self.metas = self.index.load_meta()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embedder.encode([query])
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for s, i in zip(scores, idxs):
            if i == -1:  # 保险
                continue
            meta = self.metas[i]
            results.append({"score": float(s), **meta})
        return results
```

---

# rag/reranker.py

```python
from __future__ import annotations
from typing import List, Dict

# 可选CrossEncoder；若缺失则退化为“按检索分数”
try:
    from sentence_transformers import CrossEncoder
    _HAS_CE = True
except Exception:
    _HAS_CE = False

class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model = None
        if model_name and _HAS_CE:
            try:
                self.model = CrossEncoder(model_name)
            except Exception:
                self.model = None

    def rerank(self, query: str, passages: List[Dict], top_k: int = 5) -> List[Dict]:
        if not passages:
            return []
        if self.model is None:
            # 退化：直接按初检索分数排序
            return sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
        pairs = [(query, p["text"]) for p in passages]
        scores = self.model.predict(pairs)
        for p, s in zip(passages, scores):
            p["rerank_score"] = float(s)
        return sorted(passages, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

---

# rag/generator.py

```python
from __future__ import annotations
from typing import List
import os

from .pipeline_utils import format_context

class Generator:
    def __init__(self, provider: str = "none", openai_model: str | None = None, hf_model: str | None = None):
        self.provider = provider
        self.openai_model = openai_model
        self.hf_model = hf_model
        self._init_backend()

    def _init_backend(self):
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception:
                raise RuntimeError("OpenAI 未安装或环境变量未设置")
        elif self.provider == "hf":
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self._tok = AutoTokenizer.from_pretrained(self.hf_model)
                self._mdl = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model)
            except Exception:
                raise RuntimeError("Transformers/HF 模型不可用")
        else:
            # none: 使用模板生成（最简单）
            pass

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = (
            "你是一个严谨的助手。仅根据下列资料回答问题；若无答案请说‘未找到’。\n\n" +
            format_context(contexts) +
            f"\n问题：{question}\n回答："
        )
        if self.provider == "openai":
            msg = self._client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": "你是严谨的中文助理。"},
                         {"role": "user", "content": prompt}],
                temperature=0.2
            )
            return msg.choices[0].message.content.strip()
        elif self.provider == "hf":
            import torch
            inputs = self._tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self._mdl.generate(**inputs, max_new_tokens=256)
            return self._tok.decode(out[0], skip_special_tokens=True)
        else:
            # 最简：把证据拼接成摘要式回答
            if not contexts:
                return "未找到。"
            joined = "\n\n".join(contexts[:3])
            return f"基于资料，摘要如下：\n{joined}"
```

---

# rag/pipeline_utils.py

```python
from __future__ import annotations
from typing import List

def format_context(contexts: List[str]) -> str:
    out = ["【资料】"]
    for i, c in enumerate(contexts, 1):
        out.append(f"[{i}]\n{c}")
    return "\n\n".join(out)
```

---

# rag/pipeline.py

```python
from __future__ import annotations
from typing import List, Dict

from .retriever import Retriever
from .reranker import Reranker
from .generator import Generator
from .pipeline_utils import format_context

class RAGPipeline:
    def __init__(self, index_dir: str, embed_model: str, rerank_model: str | None, gen_provider: str, openai_model: str | None, hf_model: str | None):
        self.retriever = Retriever(index_dir, embed_model)
        self.reranker = Reranker(rerank_model)
        self.generator = Generator(gen_provider, openai_model, hf_model)

    def answer(self, question: str, top_k: int = 8, rerank_top_k: int = 4) -> Dict:
        # 1) 初检索
        cands = self.retriever.retrieve(question, top_k=top_k)
        # 2) 重排（可选）
        reranked = self.reranker.rerank(question, cands, top_k=rerank_top_k)
        # 3) 取上下文并生成
        contexts = [r["text"] for r in reranked]
        answer = self.generator.generate(question, contexts)
        return {
            "question": question,
            "answer": answer,
            "contexts": reranked,
        }
```

---

# build_index.py

```python
import os, glob
from tqdm import tqdm
from rag.chunks import split_into_chunks
from rag.embedder import Embedder
from rag.index import FaissIndex
from config import settings

import json

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data/docs")
    ap.add_argument("--index_dir", type=str, default="./data/index")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()

    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(args.data_dir, ext)))
    assert files, f"未找到文本文件，请放入 {args.data_dir} 下的 .txt/.md 文件"

    embedder = Embedder(settings.EMBEDDING_MODEL)

    metas = []
    chunks_all = []
    for fp in tqdm(files, desc="分块中"):
        text = read_text_file(fp)
        chunks = split_into_chunks(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for ck in chunks:
            metas.append({
                "id": len(metas),
                "text": ck,
                "source": os.path.basename(fp),
            })
            chunks_all.append(ck)

    # 向量化
    from math import ceil
    B = 512
    vecs = []
    for i in tqdm(range(ceil(len(chunks_all)/B)), desc="向量化中"):
        batch = chunks_all[i*B:(i+1)*B]
        vecs.append(embedder.encode(batch))
    import numpy as np
    vectors = np.vstack(vecs)

    # 构建索引 + 保存元数据
    index = FaissIndex(args.index_dir)
    index.build(vectors)
    index.save_meta(metas)

    print(f"已完成索引构建：{args.index_dir}")
```

---

# query.py

```python
from config import settings
from rag.pipeline import RAGPipeline

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="你的问题")
    ap.add_argument("--index_dir", type=str, default="./data/index")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--rerank_top_k", type=int, default=4)
    ap.add_argument("--with_rerank", action="store_true")
    args = ap.parse_args()

    rerank_model = settings.RERANK_MODEL if args.with_rerank else None

    pipe = RAGPipeline(
        index_dir=args.index_dir,
        embed_model=settings.EMBEDDING_MODEL,
        rerank_model=rerank_model,
        gen_provider=settings.GEN_PROVIDER,
        openai_model=settings.OPENAI_MODEL,
        hf_model=settings.HF_GEN_MODEL,
    )

    out = pipe.answer(args.question, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
    print(json.dumps(out, ensure_ascii=False, indent=2))
```

---

# 小结与下一步

* 这是最小可用的**入门RAG骨架**：分块 → 向量化 → FAISS检索 →（可选）重排 →（可选）生成。
* 你可以逐步升级：

  1. `chunks.py` 改为按句/Markdown/标题层级分块；
  2. `index.py` 切换 `IndexFlatIP` 为 IVF/HNSW+PQ 做**量化/加速**；
  3. `retriever.py` 融合 BM25 做 **Hybrid Search**；
  4. `generator.py` 默认改为 OpenAI/HF 真实生成并加入**引用**；
  5. 加个 `FastAPI` 暴露 HTTP 接口，或做成 `Streamlit` Demo。

```
```
