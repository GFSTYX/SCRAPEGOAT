import logging
from pathlib import Path

from gfwldata.config.settings import settings
from gfwldata.utils.db import get_db_session
from gfwldata.utils.logger import setup_logger
from gfwldata.utils.models import Job, JobState
from gfwldata.utils.s3 import S3Client

setup_logger(Path("gfwldata/logs/sync_jobs_from_s3.log"))
logger = logging.getLogger("scripts.dev.sync_jobs_from_s3")


def run_pipeline():
    logger.info("Starting sync jobs from s3 pipeline")

    logger.info("Extracting files from s3")
    extracted_files = extract_files_from_s3()

    logger.info("Updating statuses of files found in jobs table")
    updated_job_status(extracted_files)

    logger.info("Sync jobs from s3 pipeline complete")


def extract_files_from_s3() -> list[str]:
    # Folder prefix to list objects from s3
    PREFIX = "replays/"
    s3_client = S3Client(config=settings, bucket_name="gfwl")
    extracted_files = s3_client.list_objects(prefix=PREFIX)

    # Remove prefix from file names
    return [f.removeprefix(PREFIX) for f in extracted_files if f != PREFIX]


def updated_job_status(extracted_files: list[str]) -> None:
    with get_db_session() as session:
        # Get all records that is already processed in s3
        jobs = session.query(Job).filter(Job.s3_key.in_(extracted_files)).all()

        if not jobs:
            logger.info("No matching jobs found for update.")
            return

        # Update their status as s3_completed
        for job in jobs:
            job.state = JobState.S3_COMPLETED

        session.commit()
        logger.info("Updated %d job(s) to s3_completed status.", len(jobs))


if __name__ == "__main__":
    run_pipeline()
