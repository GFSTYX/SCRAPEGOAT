import pandas as pd
from sqlalchemy.orm import Session

from gfwldata.utils.models import Job, JobState, LeagueMatch


class LeagueDataLoader:
    """Loads league match data from a Pandas DataFrame into the database."""

    def __init__(self, db_session: Session):
        """Initializes the LeagueDataLoader with a database session."""
        self.db_session = db_session

    def load_data(self, league_data: pd.DataFrame) -> None:
        """Loads league match data from a Pandas DataFrame into the database."""

        # Sqlite doesn't handle pd.NA, change to None
        league_data = league_data.replace({pd.NA: None})

        for row in league_data.itertuples():
            # Create LeagueMatch object
            league_match = LeagueMatch(
                season=row.season,
                week=row.week,
                team1=row.team1,
                team2=row.team2,
                team1_player=row.team1_player,
                team2_player=row.team2_player,
                team1_player_deck_type=row.team1_player_deck_type,
                team2_player_deck_type=row.team2_player_deck_type,
                match_score=row.match_score,
                replay_id=int(row.replay_id) if row.replay_id else None,
                replay_url=row.replay_url,
            )

            self.db_session.add(league_match)
            self.db_session.flush()

            # Create Job object if replay_id is not null
            if self._validate_row_for_jobs(row):
                job = Job(
                    league_match_id=league_match.id,
                    state=JobState.S3_PENDING,
                    s3_key=f"{row.replay_id}_replay.json",
                )
                self.db_session.add(job)

        # Commit the session to save all changes
        self.db_session.commit()

    def _validate_row_for_jobs(self, row: tuple) -> bool:
        """Validates row (match) to be inserted to jobs table."""

        # Jobs has replay_id as non-nullable
        if not row.replay_id:
            return False

        """
        Decent check to see if the replay has the full match.
        
        When a match_score has an astericks or parenthesis, that usually means:
            1. The match is split across many replays
            2. There's no replay for the match
            
        Jobs only process replays with full matches to not have misleading analysis later.
        """
        if "*" in row.match_score or "(" in row.match_score:
            return False

        return True
