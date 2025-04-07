from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # Database connections
    SYNC_DB_URL: str = Field(
        default="sqlite:///gfwldata/data/gfwldata.db",
        description="Synchronous database engine url",
    )
    ASYNC_DB_URL: str = Field(
        default="sqlite+aiosqlite:///gfwldata/data/gfwldata.db",
        description="Asynchronous database engine url",
    )

    # AWS authentification
    AWS_REGION: str = Field(description="Required for aws s3 authentification")
    AWS_ACCESS_KEY_ID: str = Field(description="Required for aws s3 authentification")
    AWS_SECRET_ACCESS_KEY: str = Field(
        description="Required for aws s3 authentification"
    )


settings = Settings()
