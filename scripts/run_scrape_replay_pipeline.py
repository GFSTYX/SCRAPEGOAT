import asyncio
import functools
import logging
from pathlib import Path

import aiometer
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError, async_playwright
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from tenacity import (
    RetryError,
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gfwldata.config.replay import replay_settings
from gfwldata.config.settings import settings
from gfwldata.extractors.replay_extractor import (
    ReplayExtractionError,
    ReplayExtractor,
)
from gfwldata.utils.db import get_async_db_session
from gfwldata.utils.logger import setup_logger
from gfwldata.utils.models import Job, JobState, LeagueMatch
from gfwldata.utils.s3 import AsyncS3Client, get_async_s3_session

setup_logger(Path("gfwldata/logs/run_scrape_replay_pipeline.log"))
logger = logging.getLogger("scripts.run_scrape_replay_pipeline")


async def run_pipeline():
    BUCKET_NAME = "gfwl"

    async with (
        get_async_db_session() as db_session,
        async_playwright() as playwright_client,
        get_async_s3_session(settings, BUCKET_NAME) as s3_session,
    ):
        pending_jobs = await get_pending_jobs(db_session)

        if not pending_jobs:
            logger.info("No pending jobs found. Exiting pipeline.")
            return

        extractor = ReplayExtractor(replay_settings, playwright_client)

        await aiometer.run_on_each(
            async_fn=functools.partial(
                process_job_wrapper, extractor, db_session, s3_session
            ),
            args=pending_jobs,
            max_at_once=replay_settings.AIOMETER_MAX_CONCURRENT,
            max_per_second=replay_settings.AIOMETER_MAX_PER_SECOND,
        )


async def get_pending_jobs(db_session: AsyncSession) -> list[tuple[str, str, str]]:
    """Fetches pending jobs from the database."""
    statement = (
        select(Job.league_match_id, Job.s3_key, LeagueMatch.replay_url)
        .join(LeagueMatch.jobs)
        .filter(Job.state == JobState.S3_PENDING)  # S3_FAILED if retrying
    )
    result = await db_session.execute(statement)
    pending_jobs = result.all()

    logger.info("Found %s pending jobs to process.", len(pending_jobs))
    return pending_jobs


async def process_job_wrapper(
    extractor: ReplayExtractor,
    db_session: AsyncSession,
    s3_session: AsyncS3Client,
    job: tuple[str, str, str],
):
    """Wraps process_job to handle RetryError."""
    try:
        await process_job(extractor, db_session, s3_session, job)

    except RetryError:
        job_id = job[0]
        statement = (
            update(Job)
            .where(Job.league_match_id == job_id)
            .values(state=JobState.S3_FAILED)
        )
        await db_session.execute(statement)
        await db_session.commit()

        logger.error(
            "Job %s exceeded %s retries and has been marked as failed.",
            job_id,
            replay_settings.MAX_RETRIES,
        )


@retry(
    stop=stop_after_attempt(replay_settings.MAX_RETRIES),
    wait=wait_exponential(
        multiplier=replay_settings.EXPONENTIAL_MULTIPLIER,
        min=replay_settings.EXPONENTIAL_MIN_WAIT,
        max=replay_settings.EXPONENTIAL_MAX_WAIT,
    ),
    retry=retry_if_exception_type(
        (ReplayExtractionError, PlaywrightError, TimeoutError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
)
async def process_job(
    extractor: ReplayExtractor,
    db_session: AsyncSession,
    s3_session: AsyncS3Client,
    job: tuple[str, str, str],
) -> None:
    """Processes a single job: extracts replay JSON, uploads to S3, updates the DB."""
    job_id, s3_key, replay_url = job

    replay_json = await extractor.extract_replay_json(replay_url)

    load_result = await s3_session.put_object(f"replays/{s3_key}", replay_json, "json")

    state = JobState.S3_COMPLETED if load_result else JobState.S3_PENDING
    statement = update(Job).where(Job.league_match_id == job_id).values(state=state)
    await db_session.execute(statement)
    await db_session.commit()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
