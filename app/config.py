import threading
import tomllib
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


def get_project_root() -> Path: 
    return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()

class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")

class AppConfig(BaseModel):
    llm: LLMSettings

class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Config, cls).__new__(cls)
                    cls._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._load_initial_config()
                    self._initialized = True

    def _get_config_path(self) -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config-example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("Config file not found")

    def _load_config(self):
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        
        default_settings = {
            "model": base_llm.get("model", "claude-3-5-sonnet"),
            "base_url": base_llm.get("base_url", "https://api.openai.com/v1"),
            "api_key": base_llm.get("api_key", ""),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
        }

        config_dict = {
            "llm": default_settings
        }
        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

config = Config()