import logging
import re

import pandas as pd

from gfwldata.utils.replay_url_helpers import clean_replay_url, extract_replay_id

logger = logging.getLogger(__name__)


class ExcelLeagueDataTransformer:
    def create_transformed_df(
        self, matchups_df: pd.DataFrame, deck_history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms and merges matchups and deck history dataframes."""
        # Create copies of the dataframes
        matchups_df = matchups_df.copy()
        deck_history_df = deck_history_df.copy()

        # Clean player names
        matchups_df["team1_player"] = matchups_df["team1_player"].apply(
            self._clean_player_name
        )
        matchups_df["team2_player"] = matchups_df["team2_player"].apply(
            self._clean_player_name
        )
        deck_history_df["player_name"] = deck_history_df["player_name"].apply(
            self._clean_player_name
        )

        # Remove false positives in deck history
        deck_history_df = deck_history_df.assign(
            deck_type=lambda x: x["deck_type"].apply(self._clean_deck_type_name)
        ).dropna(subset=["deck_type"])

        # Merge dataframes
        league_df = (
            matchups_df.merge(
                deck_history_df.rename(columns={"deck_type": "team1_player_deck_type"}),
                left_on=["season", "week", "team1_player"],
                right_on=["season", "week", "player_name"],
                how="left",
            )
            .drop(columns=["player_name"])
            .merge(
                deck_history_df.rename(columns={"deck_type": "team2_player_deck_type"}),
                left_on=["season", "week", "team2_player"],
                right_on=["season", "week", "player_name"],
                how="left",
            )
            .drop(columns=["player_name"])
        )

        # Apply transformation functions
        league_df = (
            league_df.assign(
                team1=lambda x: x["team1"].apply(self._clean_team_name),
                team2=lambda x: x["team2"].apply(self._clean_team_name),
                replay_id=lambda x: x["replay_url"].apply(extract_replay_id),
                replay_url=lambda x: x["replay_id"].apply(clean_replay_url),
            )
            .dropna(subset=["team1_player", "team2_player"])
            # Remove duplicates, doesn't really matter how can't confirm what's "correct" and going to parse later
            .groupby(
                ["season", "week", "team1_player", "team2_player"], group_keys=False
            )
            .apply(lambda x: x.drop_duplicates(subset=["replay_id"]))
            .sort_values(by=["season", "week"])
        )

        return league_df

    @staticmethod
    def _clean_player_name(player_name: str) -> str:
        """Cleans the player name by removing parentheses and lowercasing."""
        if not isinstance(player_name, str):
            return player_name

        # Remove anything in parentheses, eg. (60) Nicey
        cleaned_player_name = re.sub(r"\([^)]*\)", "", player_name).strip()

        # Transform to lowercase
        cleaned_player_name = cleaned_player_name.lower()

        return cleaned_player_name

    @staticmethod
    def _clean_deck_type_name(deck_type: str) -> str | None:
        """Cleans the deck type name by returning None if it starts with an asterisk."""
        if not isinstance(deck_type, str):
            return deck_type

        """
        If the deck type starts with an astericks, return None.
        Starting with an astericks usually means there was a swapped player, not a real deck type.
        """
        return deck_type if not deck_type.startswith("*") else None

    @staticmethod
    def _clean_team_name(team_name: str) -> str:
        """Cleans the team name by removing everything after the first parenthesis."""
        if not isinstance(team_name, str):
            return team_name

        # Remove everything after the first parenthesis
        cleaned_team_name = re.sub(r"\(.*", "", team_name).strip()

        return cleaned_team_name
