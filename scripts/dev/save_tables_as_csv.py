import logging
from pathlib import Path

import pandas as pd

from gfwldata.utils.db import sync_engine
from gfwldata.utils.logger import setup_logger

setup_logger(Path("gfwldata/logs/save_tables_as_csv.log"))
logger = logging.getLogger("scripts.save_tables_as_csv")

TABLES_TO_SAVE = ["league_matches", "jobs", "event_decks", "games"]


if __name__ == "__main__":
    for table in TABLES_TO_SAVE:
        csv_path = Path(f"gfwldata/data/tables/{table}.csv")
        query = f"SELECT * FROM {table}"

        results = pd.read_sql(query, sync_engine)
        results.to_csv(csv_path, index=False)

        logger.info(f"Successfully saved table '{table}' to '{csv_path}'")
