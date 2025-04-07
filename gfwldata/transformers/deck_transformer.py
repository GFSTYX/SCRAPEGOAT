import logging
from collections import Counter
from datetime import datetime
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


class DeckTransformer:
    def transform_deck_data(self, deck_data: dict, deck_id: int) -> pd.DataFrame | None:
        """Transforms deck data into a Pandas DataFrame."""

        # Deck validator
        if not self._deck_validator(deck_data):
            logger.info("deck_id %s failed validation", deck_id)
            return

        published_at = self._clean_published_at(deck_data.get("publishDate"))

        # Base deck data
        base_deck_data = {
            "published_at": published_at,
            "deck_category": deck_data.get("category"),
            "deck_class": deck_data.get("deckTypeName"),
            "deck_builder": deck_data.get("builderName"),
            "event_name": deck_data.get("eventAbbreviation"),
            "event_placement": deck_data.get("placement"),
            "url": f"https://formatlibrary.com/api/decks/{deck_id}",
        }

        # Create dataframe rows for main and side decks
        main_deck_rows = self._create_deck_rows(
            deck_data.get("main", []), "main", base_deck_data
        )
        side_deck_rows = self._create_deck_rows(
            deck_data.get("side", []), "side", base_deck_data
        )

        transformed_df = pd.DataFrame(main_deck_rows + side_deck_rows)

        logger.info("Successfully transformed deck for deck_id  %s", deck_id)
        return transformed_df

    @staticmethod
    def _deck_validator(deck_data: dict) -> bool:
        # Check if deck_data is a dict
        if not isinstance(deck_data, dict):
            logger.warning("deck_data is not a dictionary.")
            return False

        # Check if required data is in deck_data
        required_keys = ["main", "category", "deckTypeName"]
        if not all(key in deck_data for key in required_keys):
            logger.warning("deck_data is missing required keys.")
            return False

        return True

    @staticmethod
    def _clean_published_at(published_at: str) -> datetime | None:
        """Cleans and converts the published_at string to a datetime object."""
        if not isinstance(published_at, str):
            return

        return datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def _create_deck_rows(
        cards: list[dict], deck_type: Literal["main", "side"], base_deck_data: dict
    ) -> list[dict]:
        """Creates deck rows for the main and side decks."""
        # Count card occurrences
        card_counts = Counter(card.get("name") for card in cards)

        return [
            {
                "card_name": card_name,
                "card_amount": card_amount,
                "deck_type": deck_type,
                **base_deck_data,
            }
            for card_name, card_amount in card_counts.items()
        ]
