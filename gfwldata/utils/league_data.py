import pandas as pd

from gfwldata.loaders.league_data_loader import LeagueDataLoader
from gfwldata.utils.db import get_db_session


def load_league_data(league_data: pd.DataFrame) -> None:
    with get_db_session() as db_session:
        loader = LeagueDataLoader(db_session)
        loader.load_data(league_data)
