import logging
from pathlib import Path

import httpx
import pandas as pd

from gfwldata.config.discord import discord_settings
from gfwldata.extractors.discord_league_data_extractor import DiscordLeagueDataExtractor
from gfwldata.transformers.discord_league_data_transformer import (
    DiscordLeagueDataTransformer,
)
from gfwldata.utils.league_data import load_league_data
from gfwldata.utils.logger import setup_logger

setup_logger(Path("gfwldata/logs/discord_league_data_pipeline.log"))
logger = logging.getLogger("scripts.run_discord_league_data_pipeline")


def run_pipeline():
    logger.info("Starting discord league data pipeline")

    logger.info("Extracting league data messages from discord")
    league_data_messages = extract_league_data_messages()

    logger.info("Transforming discord's league data")
    league_data = transform_league_data(league_data_messages)

    logger.info("Loading league data")
    load_league_data(league_data)

    logger.info("Excel league data pipeline complete")


def extract_league_data_messages() -> list[dict]:
    with httpx.Client() as http_client:
        extractor = DiscordLeagueDataExtractor(discord_settings, http_client)
        league_data_messages = extractor.get_league_data_messages(
            min_message_id=discord_settings.MIN_MESSAGE_ID_V1
        )

        return league_data_messages


def transform_league_data(league_data_messages: list[dict]) -> pd.DataFrame:
    transformer = DiscordLeagueDataTransformer(discord_settings)
    transformed_df = transformer.create_transformed_df(league_data_messages)

    return transformed_df


if __name__ == "__main__":
    run_pipeline()
