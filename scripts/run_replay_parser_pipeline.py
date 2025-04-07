import json
import logging
import multiprocessing as mp
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy.future import select
from sqlalchemy.orm import Session

from gfwldata.config.replay_parser import replay_parser_settings
from gfwldata.config.settings import settings
from gfwldata.transformers.replay_parser import ReplayParser
from gfwldata.utils.db import get_db_session
from gfwldata.utils.logger import init_worker_logger, setup_multiproc_logger
from gfwldata.utils.models import Game, Job, JobState
from gfwldata.utils.s3 import S3Client

logger = logging.getLogger("scripts.run_replay_parser_pipeline")


def run_pipeline() -> None:
    # Setup multiprocess logger
    log_queue, listener = setup_multiproc_logger(
        log_file=Path("gfwldata/logs/run_replay_parser_pipeline.log"),
        level=logging.INFO,
    )
    listener.start()

    # Get pending jobs
    with get_db_session() as db_session:
        pending_jobs = get_pending_jobs(db_session)

    # Process jobs
    with mp.Pool(
        processes=replay_parser_settings.MP_PROCESSES,
        initializer=worker_initializer,
        initargs=(log_queue,),
    ) as pool:
        pool.map(process_job, pending_jobs)


def get_pending_jobs(db_session: Session) -> list[tuple[UUID, str]]:
    statement = (
        select(Job.league_match_id, Job.s3_key)
        .filter(Job.state == JobState.S3_COMPLETED)
        .order_by(Job.s3_key)
    )
    result = db_session.execute(statement)
    pending_jobs = result.all()

    logger.info("Found %s pending s3_keys to process.", len(pending_jobs))
    return pending_jobs


def worker_initializer(log_queue: mp.Queue) -> None:
    global global_s3_client, global_parser

    init_worker_logger(log_queue)

    global_s3_client = S3Client(settings, "gfwl")
    global_parser = ReplayParser(replay_parser_settings)

    logger.info(
        "Initialized logger, s3_client, and parser in process: %s",
        mp.current_process().name,
    )


def process_job(job: tuple[UUID, str]) -> None:
    global global_s3_client, global_parser

    league_match_id, s3_key = job

    try:
        with get_db_session() as db_session:
            logger.info("Processing league_match_id: %s", league_match_id)

            replay_data = extract_replay_from_s3(global_s3_client, s3_key)

            games_df = global_parser.parse_replay(replay_data, league_match_id)

            load_tables_to_database(db_session, league_match_id, games_df)

            logger.info("Finished processing league_match_id: %s", league_match_id)

    except Exception:
        logger.exception(
            "Error processing league_match_id: %s, s3_key: %s", league_match_id, s3_key
        )


def extract_replay_from_s3(s3_client: S3Client, s3_key: str) -> dict:
    replay_string = s3_client.get_object(f"replays/{s3_key}")

    if not replay_string:
        raise ValueError("No data for %s found in s3", s3_key)

    return json.loads(replay_string)


def load_tables_to_database(
    db_session: Session,
    league_match_id: UUID,
    games_df: pd.DataFrame,
) -> None:
    # Sqlite doesn't handle pd.NA, change to None
    games_df = games_df.replace({pd.NA: None})

    # Create Game object
    for row in games_df.itertuples():
        game = Game(
            league_match_id=league_match_id,
            played_at=row.played_at,
            player1=row.player1,
            player2=row.player2,
            player1_deck_type=row.player1_deck_type,
            player1_deck_type_confidence=round(row.player1_deck_type_confidence, 4),
            player2_deck_type=row.player2_deck_type,
            player2_deck_type_confidence=round(row.player2_deck_type_confidence, 4),
            player1_cards=row.player1_cards,
            player2_cards=row.player2_cards,
            game_number=row.game_number,
            game_winner=row.game_winner,
            went_first=row.went_first,
        )

        # Add object to sqlalchemy session
        db_session.add(game)

    # Commit changes to database
    db_session.commit()


if __name__ == "__main__":
    run_pipeline()
