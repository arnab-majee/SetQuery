from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SatQuery"
    app_version: str = "0.1.0"

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    model_load_on_startup: bool = False

    max_upload_size_mb: int = 40

    cors_origins: list[str] = ["http://localhost:3000"]

    upload_dir: str = "uploads"
    output_dir: str = "outputs"

    max_new_tokens: int = 256

    model_config = SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore" )


settings = Settings()