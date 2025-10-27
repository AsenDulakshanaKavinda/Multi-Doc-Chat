import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import logging as log
from multi_doc_chat.exception import ProjectException
from multi_doc_chat.prompts.prompt_library import PREOMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            # load llm and prompts
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PREOMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.question_prompt: ChatPromptTemplate = PREOMPT_REGISTRY[
                PromptType.CONTEXT_QUESTION.value
            ]

            # lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info(f"ConversationalRAG initialized {self.session_id}")

        except Exception as e:
            log.error(f"Failed to initialize ConversationalRAG {e}")
            raise ProjectException(e, sys)


    def load_retriever_from_faiss(
          self,
          index_path: str,
          k: int = 5,
          index_name: str = "index",
          search_type: str = "mmr",
          fetch_k: int = 20,
          lambda_mult: float = 0.5,
          search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        
        Args:
            index_path: Path to FAISS index directory
            k: Number of documents to return
            index_name: Name of the index file
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            fetch_k: Number of documents to fetch before MMR re-ranking (only for MMR)
            lambda_mult: Diversity parameter for MMR (0=max diversity, 1=max relevance)
            search_kwargs: Custom search kwargs (overrides other parameters if provided)
        """

        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            
            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_kwargs == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs = search_kwargs
            )

            self._build_lcel_chain()

            log.info(
                f"""FAISS retriever loaded successfully, 
                index_path={index_path},
                index_name={index_name},
                search_type={search_type},
                k={k},
                fetch_k={fetch_k if search_type == "mmr" else None},
                lambda_mult={lambda_mult if search_type == "mmr" else None},
                session_id={self.session_id,}"""
            )
            return self.retriever

        except Exception as e:
            log.error(f"Failed to load retriever from FAISS {str(e)}")
            raise ProjectException(e, sys)


    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        invoke the LCEL pipeline.
        """
        try:
            if self.chain is None:
                raise ProjectException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning(f"No answer generated, user_input={user_input}, session_id={self.session_id}")
                return "No answer generated."
        
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                log.error(f"Invalid chat answer {str(ve)}")
                raise ProjectException(ve, sys)
            log.info(
                """Chain invoked successfully,
                session_id={self.session_id},
                user_input={user_input},
                answer_preview={answer[:100]}""",
            )
            return answer
        except Exception as e:
            log.error(f"Failed to invoke ConversationalRAG {e}")
            raise ProjectException("Invocation error in ConversationalRAG", sys)

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error(f"Failed to load LLM {e}")
            raise ProjectException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise ProjectException("No retriever set before building chain", sys)
            
            # 1). rewrite the question with chat histroy context
            question_rewrite = (
                {"input": itemgetter("input"), "chat_histry": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser
            )

            # 2). retrive docs for rewrite questio
            retrieve_docs = question_rewrite | self.retriever | self._format_docs

            # 3). answer using retrieved context + original input + chat histroy
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_histroy": itemgetter("chat_history"),
                }
                | self.question_prompt
                | self.llm
                |StrOutputParser
            )
            log.info(f"LCEL graph built successfully, session_id={self.session_id}")
        except Exception as e:
            log.error("Failed to build LCEL chain, error={e}, session_id={self.session_id}")
            raise ProjectException("Failed to build LCEL chain", sys)
