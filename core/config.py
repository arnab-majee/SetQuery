from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_name: str = Field(default="SatQuery AI")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=True)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    frontend_url: str = Field(default="http://localhost:5173")
    upload_dir: str = Field(default="uploads")
    output_dir: str = Field(default="outputs")
    max_upload_size_mb: int = Field(default=100)
    model_name: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.2)
    model_load_on_startup: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [self.frontend_url, "http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def upload_path(self) -> Path:
        path = PROJECT_ROOT / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_path(self) -> Path:
        path = PROJECT_ROOT / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()