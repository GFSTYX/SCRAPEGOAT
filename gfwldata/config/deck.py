from pydantic_settings import BaseSettings


class DeckSettings(BaseSettings):
    # Deck extractor settings
    EXTRACTOR_BASE_URL: str

    # Http client settings
    READ_TIMEOUT: int
    CONNECT_TIMEOUT: int

    # Retry settings
    MAX_RETRIES: int
    EXPONENTIAL_MIN_WAIT: int
    EXPONENTIAL_MAX_WAIT: int
    EXPONENTIAL_MULTIPLIER: int

    # Aiometer settings
    AIOMETER_MAX_CONCURRENT: int
    AIOMETER_MAX_PER_SECOND: int


deck_settings = DeckSettings(
    EXTRACTOR_BASE_URL="https://formatlibrary.com/api/decks",
    READ_TIMEOUT=5,
    CONNECT_TIMEOUT=5,
    MAX_RETRIES=2,
    EXPONENTIAL_MIN_WAIT=2,
    EXPONENTIAL_MAX_WAIT=8,
    EXPONENTIAL_MULTIPLIER=2,
    AIOMETER_MAX_CONCURRENT=100,
    AIOMETER_MAX_PER_SECOND=50,
)
5
