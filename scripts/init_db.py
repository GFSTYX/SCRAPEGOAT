import logging
from pathlib import Path

from gfwldata.utils.db import sync_engine
from gfwldata.utils.logger import setup_logger
from gfwldata.utils.models import Base

setup_logger(Path("gfwldata/logs/init_db.log"))
logger = logging.getLogger("scripts.init_db")


def init_db():
    logger.info("Creating all database tables...")

    try:
        Base.metadata.create_all(sync_engine)
        logger.info("Successfully created all tables!")

    except Exception:
        logger.exception("Error creating tables")
        raise


if __name__ == "__main__":
    init_db()
