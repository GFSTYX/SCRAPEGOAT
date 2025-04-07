from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReplaySettings(BaseSettings):
    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # Brightdata endpoint
    SBR_WS_ENDPOINT: str = Field(
        description="Brightdata's websocket endpoint to use their Scraping Browser service",
    )

    # Replay extractor settings
    SCREENSHOT_DIR: str = Field(
        description="Directory to place screenshots of the browser when the replay extractor fails"
    )
    S3_PREFIX: str = Field(
        description="The prefix, or folder, of the replay files in s3"
    )
    PAGE_TIMEOUT: int = Field(description="Playwright's page timeout in milliseconds")

    # Retry settings
    MAX_RETRIES: int
    EXPONENTIAL_MIN_WAIT: int
    EXPONENTIAL_MAX_WAIT: int
    EXPONENTIAL_MULTIPLIER: int

    # Aiometer settings
    AIOMETER_MAX_CONCURRENT: int
    AIOMETER_MAX_PER_SECOND: int


replay_settings = ReplaySettings(
    SCREENSHOT_DIR="gfwldata/data/screenshots",
    S3_PREFIX="replays/",
    PAGE_TIMEOUT=1000 * 60,
    MAX_RETRIES=3,
    EXPONENTIAL_MIN_WAIT=2,
    EXPONENTIAL_MAX_WAIT=16,
    EXPONENTIAL_MULTIPLIER=2,
    AIOMETER_MAX_CONCURRENT=30,
    AIOMETER_MAX_PER_SECOND=2,
)
