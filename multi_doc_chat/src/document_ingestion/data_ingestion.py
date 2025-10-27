"""
Saves uploaded files,
Loads and splits them into chunks,
Converts text into embeddings,
Builds or updates a FAISS vector database,
Returns a retriever object (used later for querying documents with an LLM).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import logging as log
from multi_doc_chat.exception import ProjectException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
import hashlib
import sys

def generate_sesstion_id() -> str:
    # Create a unique folder name per user session.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}" # session_20251026_103848_269900d5

class ChatIngestor:
    def __init__(
            self,
            temp_base: str = "data",
            faiss_base: str = "faiss_index",
            use_session_dirs: bool = True,
            session_id: Optional[str] = None
    ):
        try:
            self.model_loader = ModelLoader() # create model loader
            self.user_session = use_session_dirs 
            self.session_id = session_id or generate_sesstion_id() 
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.fasiss_base = Path(faiss_base); self.fasiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.fasiss_base)

            log.info("ChatIngestor initialized")

        except Exception as e:
            log.error(f"Failed to initialize ChatIngestor {e}")
            raise ProjectException(e, sys)

    def  _resolve_dir(self, base: Path):
        """
        create and return
        session subfolder or base folder directly
        """

        if self.user_session:
            d = base / self.session_id  # "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base # fallback: "faiss_index/"
    
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        log.info(f"Documents split, chunks={len(chunks)}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        return chunks
    
    def build_retriver(
            self,
            uploaded_file: Iterable,
            *,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            k: int = 5,
            search_type: str = "mmr",
            fetch_k: int = 20,
            lambda_mult: float = 0.5
        ):
        try:
            paths = save_uploaded_files(uploaded_file, self.temp_dir) # save uploaded files in disk
            docs = load_documents(paths) # read files
            chunks = self._split(docs) # split docs to chunks

            fm = FaissManager(self.faiss_dir, self.model_loader) # faiss mamager
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas) # vecterstore
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas) # vecterstore

            added = fm.add_documents(chunks) # add new embeddings
            log.info(f"FAISS index updated {str(self.faiss_dir)}")

            # Configure search parameters based on search type
            search_kwargs = {"k": k}

            if search_type == "mmr":
                # MMR needs fetch_k (docs to fetch) and lambda_mult (diversity parameter)
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                # log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
                log.info("Using MMR search")

            return vs._as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error(f"Faild to build retriver {e}")
            raise ProjectException(e, sys)
_split

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# FAISS Manager (load-or-create)
class FaissManager:
    # Responsible for managing FAISS index creation, updating, and deduplication.
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str: Any] = {"rows": {}} # this is dict of rows

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists

        self.medel_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None


    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Creates a unique identifier for each document chunk based on:
            File path, orRow ID, orHash of the text content.
        Used to avoid re-adding duplicate chunks.
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("raw_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
         self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]):
        """
        Adds only new chunks:
            Generates fingerprints.
            Skips duplicates.
            Updates FAISS index and metadata JSON.
        """
        if self.vs in None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")

        new_docs = List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["raws"]:
                continue
            self._meta["raws"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)


    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True
            )
            return self.vs
        
        if not texts:
            raise ProjectException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs



