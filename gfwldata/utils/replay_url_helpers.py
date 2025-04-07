import re
from urllib.parse import ParseResult, parse_qs, urlparse

import pandas as pd


def extract_replay_id(replay_url: str) -> pd.Int64Dtype:
    """Extracts the replay ID from the replay URL."""
    parsed_url = urlparse(replay_url)
    query_params = parse_qs(parsed_url.query)

    if not validate_replay_url(parsed_url, query_params):
        return pd.NA

    # Remove user id prefix from replay's id
    replay_id = int(re.sub(r"^\d+-", "", query_params["id"][0]))

    # Return as Int64
    return pd.Series([replay_id], dtype="Int64")[0]


def validate_replay_url(parsed_url: ParseResult, query_params: dict) -> bool:
    """Validates the replay URL."""
    # Check if hostname is duelingbook
    if parsed_url.hostname != "www.duelingbook.com":
        return False

    # Check if it has an id parameter
    if "id" not in query_params:
        return False

    # Remove user id prefix from replay's id
    replay_id = re.sub(r"^\d+-", "", query_params["id"][0])

    # Check if id is an integer
    if not replay_id.isdigit():
        return False

    # Check if id has less than 15 digits
    if len(replay_id) > 15:
        return False

    return True


def clean_replay_url(replay_id: int) -> str | None:
    """Cleans the replay URL by creating a valid duelingbook URL."""
    if pd.isna(replay_id):
        return pd.NA

    return f"https://duelingbook.com/replay?id={replay_id:.0f}"
