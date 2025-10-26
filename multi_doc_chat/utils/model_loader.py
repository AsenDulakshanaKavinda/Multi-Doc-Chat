
import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from multi_doc_chat.logger import logging as log
from multi_doc_chat.exception import ProjectException

class ApiKeyManage:
    # Handles loading and validation of all required API keys
    REQUIRED_KEYS = ["MISTRAL_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("MISTRAL_API_KEY")

        # if api key is there make a dict and pass it to self.api_keys
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaede API KEYS from ECS secret.")
            except Exception as e:
                log.warning("Failed to parse API KEYS as JSON")

        # if keys not in api_keys that in main, read env and load the to self.api_keys
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # check for missing keys
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error(f"missing required API keys {missing}")
            raise ProjectException(f"Missing  {missing} API KEY", sys)

        # log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})
        log.info("API keys loaded")

    # return api keys
    def get(self, key:str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing.")
        return val
        

class ModelLoader():
    # Loads embedding models and LLMs based on config and environment.
    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded.")
        else:
            log.info("Running in PRODUCION mode.")

        self.api_key_mgt = ApiKeyManage()
        self.config = load_config()
        # log.info("YAML config loaded", config_keys=list(self.config.keys()))
        log.info("YAML config loaded")


    def load_llm(self):
        # load and return the congigured LLM model

        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "mistral")

        if provider_key not in llm_block:
            # log.error("LLM provider not found in config", provider=provider_key)
            log.error("LLM provider not found in config")
            raise ValueError(f"LLM provider '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        # log.info("Loading LLM", provider=provider, model=model_name)
        log.info("Loading LLM")

        if provider == "mistral":
            return ChatMistralAI(
                model=model_name,
                mistral_api_key=self.api_key_mgt.get("MISTRAL_API_KEY"),
                temperature=temperature,
                model_kwargs={"max_output_tokens": max_tokens}
            )
        else:
            # log.error("Unsupported LLM provider", provider=provider)
            log.error("Unsupported LLM provider")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def load_embeddings(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model")
            
            return MistralAIEmbeddings(model=model_name,
                                       mistral_api_key=self.api_key_mgt.get("MISTRAL_API_KEY"))
        except Exception as e:
            # todo: change this
            # log.error("Error loading embedding model", error=str(e))
            # raise DocumentPortalException("Failed to load embedding model", sys)
            log.error("Error loading embedding model")
            ProjectException(e, sys)

def test():
    loader = ModelLoader()
    loader.load_embeddings()
    loader.load_llm()





