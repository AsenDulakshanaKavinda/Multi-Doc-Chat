from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import logging as log
from multi_doc_chat.exception import ProjectException
from fastapi import UploadFile
import sys

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load docs using appropriate loader based on extension.
    """

    docs: list[Document] = []

    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning(f"Unsupported extension skipped {p}")
                continue
            docs.extend(loader.load())
        log.info(f"{len(docs)} Documents loaded")

    except Exception as e:
        log.error(f"Failed loading documents {e}")
        raise ProjectException(e, sys)
            

class FastAPIFileAdapter:
    """
    Adapt FastAPI UploadFile to a simple object with .name and .getbuffer().
    """

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()






