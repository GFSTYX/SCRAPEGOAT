import logging
import re
from uuid import UUID

import joblib
import pandas as pd

from gfwldata.config.replay_parser import ReplayParserSettings

logger = logging.getLogger(__name__)


class ReplayParser:
    def __init__(self, config: ReplayParserSettings):
        self.config = config
        self.label_encoder = joblib.load(self.config.LABEL_ENCODER_PATH)
        self.vectorizer = joblib.load(self.config.VECTORIZER_PATH)
        self.model = joblib.load(self.config.MODEL_PATH)

    def parse_replay(self, replay_data: dict, league_match_id: UUID) -> pd.DataFrame:
        """Parse replay data and return a DataFrame of game results."""
        if not self._validate_replay_data(replay_data):
            logger.error("league_match_id %s replay failed to parse", league_match_id)
            return

        plays_df = self._create_plays_df(replay_data)

        plays_df = plays_df.assign(
            card_name=lambda df: df.apply(self._extract_card_name, axis=1),
            deck_change=lambda df: df.apply(self._calculate_deck_change, axis=1),
            game_number=lambda df: df.public_log.str.contains("Chose to go").cumsum(),
        )

        played_at = pd.to_datetime(replay_data.get("date"))
        player1 = replay_data.get("player1").get("username")
        player2 = replay_data.get("player2").get("username")

        games_df = self._create_games_df(played_at, player1, player2, plays_df)

        logger.info("Replay parse for league_match_id %s is complete", league_match_id)
        return games_df

    def _validate_replay_data(self, replay_data: dict) -> bool:
        """Validate that replay_data is a dict and contains necessary keys."""
        if not isinstance(replay_data, dict):
            logger.warning("replay_data is not a dict")
            return False

        if "plays" not in replay_data:
            logger.warning("replay_data does not contain plays key")
            return False

        return True

    def _create_plays_df(self, replay_data: dict) -> pd.DataFrame:
        """Create plays_df from replay_data's plays log."""
        plays = []

        for play in replay_data.get("plays"):
            base = {
                "seconds": play.get("seconds"),
                "play": play.get("play"),
                "owner": play.get("owner"),
            }

            logs = play.get("log")

            if isinstance(logs, list):
                for log in logs:
                    plays.append({**base, **log})
            elif isinstance(logs, dict):
                plays.append({**base, **logs})

        return (
            pd.json_normalize(plays)
            .assign(username=lambda df: df["owner"].fillna(df["username"]))
            .drop(columns="owner")
        )

    def _extract_card_name(self, row: pd.Series) -> str | None:
        """Extract card name from logs."""
        # Don't extract cards from messages
        if row.play == "Duel message":
            return None

        for log in (row.private_log, row.public_log):
            if not log:
                continue

            matches = re.findall(r'"([^"]*)"', str(log))
            if matches:
                return matches[0]

        return None

    def _calculate_deck_change(self, row: pd.Series) -> int:
        """Determine whether cards were added or returned to deck."""
        logs = [str(row.private_log), str(row.public_log)]

        # Card added from deck
        if any(
            phrase in log
            for log in logs
            for phrase in ("Drew", "from Deck", "from top of deck")
        ):
            return 1

        # Card returned to deck
        if any(
            phrase in log
            for log in logs
            for phrase in ("to top of deck", "to bottom of deck")
        ):
            return -1

        return 0

    def _create_games_df(
        self,
        played_at: pd.Timestamp,
        player1: str,
        player2: str,
        plays_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create games_df incorporating plays and deck type predictions."""
        games_data = []
        games_base = {"played_at": played_at, "player1": player1, "player2": player2}

        # Get match cards df
        cards_df = self._create_cards_df(plays_df)

        for game in range(1, max(plays_df["game_number"]) + 1):
            # Filter plays_df by game_number
            game_df = plays_df.query("game_number == @game")

            # Get player cards data
            player1_cards_df = cards_df.query(
                "game_number == @game & username == @player1"
            )[["card_name", "card_amount"]]

            player2_cards_df = cards_df.query(
                "game_number == @game & username == @player2"
            )[["card_name", "card_amount"]]

            # Deck type model results
            player1_prediction = self._predict_deck_type(player1_cards_df)
            player2_prediction = self._predict_deck_type(player2_cards_df)

            games_data.append(
                {
                    **games_base,
                    "game_number": game,
                    "game_winner": self._get_game_winner(player1, player2, game_df),
                    "went_first": game_df.query("public_log == 'Chose to go first'")[
                        "username"
                    ].item(),
                    "player1_cards": player1_cards_df.to_dict("records"),
                    "player2_cards": player2_cards_df.to_dict("records"),
                    "player1_deck_type": player1_prediction[0],
                    "player1_deck_type_confidence": player1_prediction[1],
                    "player2_deck_type": player2_prediction[0],
                    "player2_deck_type_confidence": player2_prediction[1],
                }
            )

        return pd.DataFrame(games_data)

    def _create_cards_df(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """Create cards_df with cumulative deck changes into a DataFrame."""
        return (
            plays_df.dropna(subset="card_name")
            .assign(
                cum_deck_change=lambda df: df.groupby(
                    ["game_number", "username", "card_name"]
                )["deck_change"].cumsum()
            )
            .groupby(["game_number", "username", "card_name"])
            .agg(card_amount=("cum_deck_change", "max"))
            .reset_index()
            .query("card_amount > 0")
        )

    def _get_game_winner(
        self, player1: str, player2: str, game_df: pd.DataFrame
    ) -> str | None:
        """Determine the game winner based on defeat logs."""
        # Get game loser if one exists
        game_loser = game_df.query("public_log in ['Admitted defeat', 'Lost Duel']")[
            "username"
        ]

        # If there's no game loser, then there's a draw
        if game_loser.empty:
            return None

        # Continue with returning game winner if no draw
        return player1 if game_loser.item() == player2 else player2

    def _predict_deck_type(self, player_cards_df: pd.DataFrame) -> tuple[str, float]:
        """Predict deck type for a player's cards using the deck type model."""
        # Repeat card_name by card_amount, then turn to list, then concat by "|" separator
        cards_str = "|".join(
            player_cards_df["card_name"].repeat(player_cards_df["card_amount"]).tolist()
        )

        # Model prediction
        vectorized_data = self.vectorizer.transform([cards_str])
        prediction = self.model.predict(vectorized_data)
        predicted_deck_type = self.label_encoder.inverse_transform(prediction)[0]

        # Get max prediction probability as confidence
        prediction_probability = self.model.predict_proba(vectorized_data)
        confidence = max(prediction_probability[0])

        return predicted_deck_type, round(confidence, 4)
