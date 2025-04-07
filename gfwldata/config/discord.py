from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DiscordSettings(BaseSettings):
    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow", env_prefix="DISCORD_"
    )

    # Request headers
    HEADER_REFERER: str
    HEADER_USER_AGENT: str
    HEADER_AUTHORIZATION: str
    HEADER_SUPER_PROPERTIES: str

    # Url params
    PARAM_CHANNEL_ID: str = Field(
        description="The id of the channel to filter messages - eg. #match-results"
    )
    PARAM_AUTHOR_ID: str = Field(
        description="The id of the author to filter messages - eg. LeagueBot"
    )

    # Request settings
    REQUEST_BASE_URL: str = Field(description="Base url for requests")
    REQUEST_PAGE_SIZE: int = Field(
        default=25, description="The number of messages per page"
    )
    REQUEST_DELAY: int = Field(
        default=1, description="The number of seconds to throttle requests in seconds"
    )

    # NOTE: Discord's message ids stores timestamp information
    MIN_MESSAGE_ID_V1: int = Field(
        description="Starting from this message id, embeds are in v1 format"
    )
    MIN_MESSAGE_ID_V2: int = Field(
        description="Starting from this message id, embeds and components are in v2 format"
    )


discord_settings = DiscordSettings(
    MIN_MESSAGE_ID_V1=1322578203826458686,
    MIN_MESSAGE_ID_V2=1326833978522599477,
)
