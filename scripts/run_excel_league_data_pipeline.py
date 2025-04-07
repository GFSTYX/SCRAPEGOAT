import logging
from pathlib import Path

import pandas as pd

from gfwldata.config.worksheet import worksheet_settings
from gfwldata.extractors.excel_league_data_extractor import ExcelLeagueDataExtractor
from gfwldata.transformers.excel_league_data_transformer import (
    ExcelLeagueDataTransformer,
)
from gfwldata.utils.league_data import load_league_data
from gfwldata.utils.logger import setup_logger

setup_logger(Path("gfwldata/logs/excel_league_data_pipeline.log"))
logger = logging.getLogger("scripts.run_excel_league_data_pipeline")


def run_pipeline():
    logger.info("Starting excel league data pipeline")

    logger.info("Extracting league data from excel")
    matchups_data, deck_history_data = extract_league_data()

    logger.info("Transforming excel's league data")
    league_data = transform_league_data(matchups_data, deck_history_data)

    logger.info("Loading league data")
    load_league_data(league_data)

    logger.info("Excel league data pipeline complete")


def extract_league_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_matchups_data = []
    all_deck_history_data = []

    for config in worksheet_settings:
        try:
            logger.info("Processing league data for %s", config.FILEPATH)
            extractor = ExcelLeagueDataExtractor(config)

            matchups_data, deck_history_data = extractor.get_league_data()
            all_matchups_data.append(matchups_data)
            all_deck_history_data.append(deck_history_data)

        except Exception:
            logger.exception("Failed to process league data for %s", config.FILEPATH)

    # Combine results from all seasons
    combined_matchups_data = pd.concat(all_matchups_data, ignore_index=True)
    combined_deck_history_data = pd.concat(all_deck_history_data, ignore_index=True)

    return combined_matchups_data, combined_deck_history_data


def transform_league_data(
    matchups_data: pd.DataFrame, deck_history_data: pd.DataFrame
) -> pd.DataFrame:
    transformer = ExcelLeagueDataTransformer()
    transformed_data = transformer.create_transformed_df(
        matchups_data, deck_history_data
    )

    return transformed_data


if __name__ == "__main__":
    run_pipeline()
