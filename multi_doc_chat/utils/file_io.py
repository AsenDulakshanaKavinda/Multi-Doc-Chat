
from __future__ import annotations
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable, List
from multi_doc_chat.exception import ProjectException
from multi_doc_chat.logger import logging as log

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    save uploaded files and return local paths
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path]  = []
        for uf in uploaded_files:
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            ext = Path(name).suffix.lower()

            # manage unsupported files
            if ext not in SUPPORTED_EXTENSIONS:
                log.warrning(f"Unsupported file skipped {name}")
                continue

            # clean file name
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            out = target_dir / fname

            with open(out, "wb") as f:
                # Prefer underlying file buffer when available (e.g., Starlette UploadFile.file)
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    f.write(uf.file.read())
                elif hasattr(uf, "read"):
                    data = uf.read()
                    # If a memoryview is returned, convert to bytes; otherwise assume bytes
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    f.write(data)
                else:
                    # Fallback for objects exposing a getbuffer()
                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object; no readable interface")
            saved.append(out)
            log.info(f"File saved for ingestion, uploaded={name}, saved_as={str(out)}")

    except Exception as e:
        log.error("")
        raise ProjectException(e, sys)














