import re

import pandas as pd

from gfwldata.config.discord import DiscordSettings
from gfwldata.utils.replay_url_helpers import clean_replay_url, extract_replay_id


class DiscordLeagueDataTransformer:
    def __init__(self, config: DiscordSettings):
        """Initialize with Discord settings."""
        self.SEASON_NUMBER = 6
        self.MIN_MESSAGE_ID_V1 = config.MIN_MESSAGE_ID_V1
        self.MIN_MESSAGE_ID_V2 = config.MIN_MESSAGE_ID_V2

    def create_transformed_df(self, league_data_messages: list[dict]) -> pd.DataFrame:
        """Create a transformed league matchups DataFrame from Discord messages."""
        league_data = []

        for message in league_data_messages:
            message_id = int(message.get("id"))
            transformed_df = {}

            # Checks if message is in v1 format
            if self.MIN_MESSAGE_ID_V1 <= message_id < self.MIN_MESSAGE_ID_V2:
                transformed_df = self._parse_message_v1(message)

            # Checks if message is in v2 format
            if message_id >= self.MIN_MESSAGE_ID_V2:
                transformed_df = self._parse_message_v2(message)

            if transformed_df:
                transformed_df["message_id"] = message_id
                league_data.append(transformed_df)

        # Create league dataframe
        league_df = (
            pd.DataFrame(league_data)
            # Apply transformations
            .assign(
                replay_id=lambda x: x["replay_url"].apply(extract_replay_id),
                replay_url=lambda x: x["replay_id"].apply(clean_replay_url),
            )
            # The v1 parser sometimes returns "question" as team name, to replace with None
            .replace({"team1": "question", "team2": "question"}, None)
            .dropna(subset=["team1_player", "team2_player"])
            # When duplicates, keep most recent (updated) result
            .sort_values(by=["message_id"])
            .groupby(["week", "team1_player", "team2_player"], group_keys=False)
            .apply(lambda x: x.drop_duplicates(subset=["replay_id"], keep="last"))
            # Drop message_id column
            .drop(columns=["message_id"])
            # Sort
            .sort_values(by=["season", "week"])
        )

        return league_df

    def _parse_message_v1(self, message: dict) -> dict:
        """Parse a Discord message in v1 format."""
        embed: dict = message.get("embeds")[0]
        title = embed.get("title")

        if title == "Match Result Reported!":
            return self._parse_match_result_reported_v1(embed)

        if title == "Match Result Updated!":
            return self._parse_match_result_updated_v1(embed)

        return {}

    def _parse_match_result_reported_v1(self, embed: dict) -> dict:
        """Parse a 'Match Result Reported' embed in v1 format."""
        # Pattern to parse the matchup field's value
        pattern = re.compile(
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*\*\*\[(\d+)]\*\*\s*"
            r"([^(]+?)\s*\(([^)]+)\)"
            r"\n"
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*\*\*\[(\d+)]\*\*\s*"
            r"([^(]+?)\s*\(([^)]+)\)",
            re.DOTALL,
        )

        # Embed fields
        matchup_field: dict = embed.get("fields")[0]
        replay_field: dict = embed.get("fields")[1]

        # Parse matchup field
        matchup_value = matchup_field.get("value")
        match = pattern.search(matchup_value)

        # League data values
        week_number = self._parse_week_number(matchup_field.get("name"))
        # TODO: Sometimes "question" is returned for team name, to replace with None
        team1, team2 = self._get_team_names(match, [1, 2, 7, 8])
        team1_score = int(match.group(4))
        team2_score = int(match.group(10))
        match_score = f"{team1_score}-{team2_score}"
        replay_url = replay_field.get("value")

        return {
            "season": self.SEASON_NUMBER,
            "week": week_number,
            "team1": team1,
            "team1_player": match.group(5).strip(),
            "team1_player_deck_type": match.group(6).strip(),
            "team2": team2,
            "team2_player": match.group(11).strip(),
            "team2_player_deck_type": match.group(12).strip(),
            "match_score": match_score,
            "replay_url": replay_url,
        }

    def _parse_match_result_updated_v1(self, embed: dict) -> dict:
        """Parse a 'Match Result Updated' embed in v1 format."""
        # Pattern to parse the matchup field's value
        pattern = re.compile(
            r"\*\*Regular Season:\s*(Week\s*\d+)\*\*.*?"
            r"__Updated Results__\s*"
            r":mag:\s*\[\*\*(\d+)\*\*]\s*([^\n]+)\s*"
            r":mag:\s*\[\*\*(\d+)\*\*]\s*([^\n]+)\s*"
            r"\*\*Replay\*\*\s*"
            r"(https://www.duelingbook.com/replay\?id=[^\s\n]+)",
            re.DOTALL | re.IGNORECASE,
        )

        # Parse embed description (combines matchup and replay fields)
        embed_description = embed.get("description")
        match = pattern.search(embed_description)

        # League data values
        week_string = match.group(1).strip()
        week_number = self._parse_week_number(week_string)
        team1_score = int(match.group(2))
        team2_score = int(match.group(4))
        match_score = f"{team1_score}-{team2_score}"

        return {
            "season": self.SEASON_NUMBER,
            "week": week_number,
            "team1": None,
            "team1_player": match.group(3).strip(),
            "team1_player_deck_type": None,
            "team2": None,
            "team2_player": match.group(5).strip(),
            "team2_player_deck_type": None,
            "match_score": match_score,
            "replay_url": match.group(6).strip(),
        }

    def _parse_message_v2(self, message: dict) -> dict:
        """Parse a Discord message in v2 format."""
        # Components have the replays in v2 format
        components = message.get("components")
        embed: dict = message.get("embeds")[0]
        embed_description = embed.get("description")

        if not self._validate_components_v2(embed, components):
            return {}

        # Description is a better validation than title in v2
        if "Updated Results" in embed_description:
            return self._parse_match_result_updated_v2(embed, components)

        else:
            return self._parse_match_result_reported_v2(embed, components)

    def _validate_components_v2(self, embed: dict, components: list[dict]) -> bool:
        """Validate components in v2 format."""
        # If there's multiple components, then the match is split across replays
        if len(components) >= 2:
            return False

        # Only want in-season matches, removing wildcards and playoffs
        if "Wildcard" in embed.get("description"):
            return False

        return True

    def _parse_match_result_reported_v2(
        self, embed: dict, components: list[dict]
    ) -> dict:
        """Parse a 'Match Result Reported' embed in v2 format."""
        # Pattern to parse the embeded matchup's description
        pattern = re.compile(
            r"\*\*(Week\s*\d+)\*\*\n"
            r"\*\*(\d+)\*\*\s*"
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*(.*?)\s*\(([^)]+)\)\n"
            r"\*\*(\d+)\*\*\s*"
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*(.*?)\s*\(([^)]+)\)",
            re.DOTALL | re.IGNORECASE,
        )

        # Find matches
        match = pattern.search(embed.get("description"))

        # Get data from embed description
        week_string = match.group(1).strip()
        week_number = self._parse_week_number(week_string)
        team1, team2 = self._get_team_names(match, [3, 4, 9, 10])
        team1_score = int(match.group(2))
        team2_score = int(match.group(8))
        match_score = f"{team1_score}-{team2_score}"

        # Get data from components
        replay_url = (
            components[0].get("components")[0].get("url")
            if len(components) == 1
            else None
        )

        return {
            "season": self.SEASON_NUMBER,
            "week": week_number,
            "team1": team1,
            "team1_player": match.group(6).strip(),
            "team1_player_deck_type": match.group(7).strip(),
            "team2": team2,
            "team2_player": match.group(12).strip(),
            "team2_player_deck_type": match.group(13).strip(),
            "match_score": match_score,
            "replay_url": replay_url,
        }

    def _parse_match_result_updated_v2(
        self, embed: dict, components: list[dict]
    ) -> dict:
        """Parse a 'Match Result Updated' embed in v2 format."""
        # Pattern to parse the embeded matchup's description
        pattern = re.compile(
            r"\*\*(Week\s*\d+)\*\*.*?"
            r"__Updated Results__\s*"
            r"\*\*(\d+)\*\*\s*"
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*([^(]+?)\s*\(([^)]+)\)\s*"
            r"\*\*(\d+)\*\*\s*"
            r"(?:<:([^:]+):\d+>|:([\w_]+):|(:question:))"
            r"\s*([^(]+?)\s*\(([^)]+)\)",
            re.DOTALL | re.IGNORECASE,
        )

        # Find matches
        match = pattern.search(embed.get("description"))

        # Get data from embed description
        week_string = match.group(1).strip()
        week_number = self._parse_week_number(week_string)
        team1, team2 = self._get_team_names(match, [3, 4, 9, 10])
        team1_score = int(match.group(2))
        team2_score = int(match.group(8))
        match_score = f"{team1_score}-{team2_score}"

        # Get data from components
        replay_url = (
            components[0].get("components")[0].get("url")
            if len(components) == 1
            else None
        )

        return {
            "season": self.SEASON_NUMBER,
            "week": week_number,
            "team1": team1,
            "team1_player": match.group(6).strip(),
            "team1_player_deck_type": match.group(7).strip(),
            "team2": team2,
            "team2_player": match.group(12).strip(),
            "team2_player_deck_type": match.group(13).strip(),
            "match_score": match_score,
            "replay_url": replay_url,
        }

    @staticmethod
    def _parse_week_number(week_str: str) -> int:
        """Parse the week number from a string."""
        match = re.search(r"Week\s*(\d+)", week_str, re.IGNORECASE)
        return int(match.group(1))

    @staticmethod
    def _get_team_names(match: re.Match, positions: list[int]) -> tuple:
        """Get team names from a regex match."""
        # Find team1 name
        if match.group(positions[0]):
            team1_name = match.group(positions[0]).strip()  # Format <:name:id>
        elif match.group(positions[1]):
            team1_name = match.group(positions[1]).strip()  # Format :name:
        else:
            team1_name = None  # Format :question:

        # Find team2 name
        if match.group(positions[2]):
            team2_name = match.group(positions[2]).strip()  # Format <:name:id>
        elif match.group(positions[3]):
            team2_name = match.group(positions[3]).strip()  # Format :name:
        else:
            team2_name = None  # Format :question:

        return team1_name, team2_name
