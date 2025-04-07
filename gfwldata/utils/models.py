import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy import (
    Enum as SQLAlchemyEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class JobState(Enum):
    # S3 states
    S3_PENDING = "s3_pending"
    S3_IN_PROGRESS = "s3_in_progress"
    S3_COMPLETED = "s3_completed"
    S3_FAILED = "s3_failed"

    # Parser states
    PARSER_PENDING = "parser_pending"
    PARSER_IN_PROGRESS = "parser_in_progress"
    PARSER_COMPLETED = "parser_completed"
    PARSER_FAILED = "parser_failed"


class BaseModel(Base):
    __abstract__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )


class LeagueMatch(BaseModel):
    __tablename__ = "league_matches"
    __table_args__ = {"comment": "League matches as defined in the excel sheets."}

    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    team1 = Column(String)
    team2 = Column(String)
    team1_player = Column(String)
    team2_player = Column(String)
    team1_player_deck_type = Column(String)
    team2_player_deck_type = Column(String)
    match_score = Column(
        String,
        comment="Score in the format 'X-Y' where X is team1 score and Y is team2 score.",
    )
    replay_id = Column(Integer)
    replay_url = Column(String)

    # Relationships
    jobs = relationship("Job", back_populates="league_matches")
    games = relationship("Game", back_populates="league_matches")

    def __repr__(self):
        return f"<LeagueMatch(season={self.season}, week={self.week}, {self.team1} vs {self.team2})>"


class Job(BaseModel):
    __tablename__ = "jobs"
    __table_args__ = {"comment": "Tracks pipeline processing state for replays."}

    league_match_id = Column(
        UUID(as_uuid=True), ForeignKey("league_matches.id"), nullable=False
    )
    state = Column(
        SQLAlchemyEnum(JobState),
        nullable=False,
        default=JobState.S3_PENDING,
        comment="Job state for different pipelines.",
    )
    s3_key = Column(
        String,
        comment="The filename, or key, of job in aws s3. Doesn't include any folder (prefix) names.",
    )

    # Relationships
    league_matches = relationship("LeagueMatch", back_populates="jobs")

    def __repr__(self):
        return f"<Job(id={self.id}, state={self.state})>"


class EventDeck(BaseModel):
    __tablename__ = "event_decks"
    __table_args__ = {"comment": "Event decks from Formatlibrary."}

    published_at = Column(
        DateTime,
        nullable=False,
        comment="Datetime when the deck was published on Formatlibrary.",
    )
    deck_type = Column(
        String,
        nullable=False,
        comment="Deck section type (e.g., 'main' or 'side').",
    )
    deck_category = Column(
        String,
        nullable=False,
        comment="Category of the deck (e.g., Aggro, Control, etc.).",
    )
    deck_class = Column(
        String,
        nullable=False,
        comment="Deck class or archetype (e.g., Warrior, Chaos Turbo).",
    )
    card_name = Column(
        String,
        nullable=False,
        comment="Name of the card included in the deck.",
    )
    card_amount = Column(
        Integer,
        nullable=False,
        comment="Number of copies of the card in this deck section.",
    )
    deck_builder = Column(String)
    event_name = Column(String)
    event_placement = Column(Integer)
    url = Column(String, nullable=False)

    def __repr__(self):
        return f"<EventDeck(event='{self.event_name}', deck_class='{self.deck_class}', builder='{self.deck_builder}')>"


class Game(BaseModel):
    __tablename__ = "games"
    __table_args__ = {"comment": "Game data from replays."}

    league_match_id = Column(
        UUID(as_uuid=True), ForeignKey("league_matches.id"), nullable=False
    )
    played_at = Column(
        DateTime, nullable=False, comment="Datetime when the duel was played."
    )
    player1 = Column(
        String, nullable=False, comment="As defined in duelingbook replays."
    )
    player2 = Column(
        String, nullable=False, comment="As defined in duelingbook replays."
    )
    player1_deck_type = Column(
        String,
        nullable=False,
        comment="Modeled from cards seen in the replay (eg. Chaos Turbo)",
    )
    player1_deck_type_confidence = Column(
        Float, nullable=False, comment="Deck model's confidence of prediction"
    )
    player2_deck_type = Column(
        String,
        nullable=False,
        comment="Modeled from cards seen in the replay (eg. Warrior)",
    )
    player2_deck_type_confidence = Column(
        Float, nullable=False, comment="Deck model's confidence of prediction"
    )
    player1_cards = Column(
        JSON,
        nullable=False,
        comment="A list of dicts that contains the card names and their amounts.",
    )
    player2_cards = Column(
        JSON,
        nullable=False,
        comment="A list of dicts that contains the card names and their amounts.",
    )
    game_number = Column(
        Integer, nullable=False, comment="The game number from a 2/3 match (eg. 1, 2)"
    )
    game_winner = Column(String, comment="The player who won the game.")
    went_first = Column(String, comment="The player who went first.")

    # Relationships
    league_matches = relationship("LeagueMatch", back_populates="games")

    def __repr__(self):
        return (
            f"<Game(player1={self.player1}, player2={self.player2}, "
            f"game_number={self.game_number}, game_winner={self.game_winner})>"
        )
