# test_chat_ingestor.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor, FaissManager

@pytest.fixture
def fake_doc():
    from langchain_core.documents import Document
    return Document(page_content="Hello World", metadata={"source": "fake.txt"})

@pytest.fixture
def fake_files(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document.")
    return [file_path]

# ---- Test 1: ChatIngestor initialization ----
def test_chat_ingestor_init(tmp_path):
    ingestor = ChatIngestor(temp_base=tmp_path, faiss_base=tmp_path)
    assert ingestor.temp_base.exists()
    assert ingestor.fasiss_base.exists()
    assert "session_" in ingestor.session_id

# ---- Test 2: Document splitting ----
def test_split_documents(fake_doc):
    ingestor = ChatIngestor()
    chunks = ingestor._split([fake_doc], chunk_size=10, chunk_overlap=0)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(hasattr(c, "page_content") for c in chunks)

# ---- Test 3: FAISS manager fingerprinting ----
def test_fingerprint_stable(fake_doc):
    fp1 = FaissManager._fingerprint(fake_doc.page_content, fake_doc.metadata)
    fp2 = FaissManager._fingerprint(fake_doc.page_content, fake_doc.metadata)
    assert fp1 == fp2  # deterministic
    assert isinstance(fp1, str)

# ---- Test 4: build_retriever pipeline (mocked) ----
@patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
@patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
@patch("multi_doc_chat.src.document_ingestion.data_ingestion.FaissManager")


def test_build_retriever(mock_faiss_manager, mock_load_docs, mock_save_files, fake_doc, fake_files):
    ingestor = ChatIngestor()

    # --- mock dependencies ---
    mock_save_files.return_value = fake_files
    mock_load_docs.return_value = [fake_doc]

    mock_vs = MagicMock()
    mock_vs._as_retriever.return_value = "mock_retriever"

    mock_faiss_instance = MagicMock()
    mock_faiss_instance.load_or_create.return_value = mock_vs
    mock_faiss_instance.add_documents.return_value = 1
    mock_faiss_manager.return_value = mock_faiss_instance

    retriever = ingestor.build_retriver(uploaded_file=fake_files)

    # --- verify behavior ---
    mock_save_files.assert_called_once()
    mock_load_docs.assert_called_once()
    mock_faiss_instance.load_or_create.assert_called_once()
    mock_faiss_instance.add_documents.assert_called_once()
    assert retriever == "mock_retriever"



