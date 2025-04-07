import asyncio
import functools
import logging
from pathlib import Path

import aiometer
import httpx
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    RetryError,
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gfwldata.config.deck import deck_settings
from gfwldata.extractors.fl_deck_extractor import FLDeckExtractor
from gfwldata.transformers.deck_transformer import DeckTransformer
from gfwldata.utils.db import get_async_db_session
from gfwldata.utils.logger import setup_logger
from gfwldata.utils.models import EventDeck

setup_logger(Path("gfwldata/logs/run_scrape_deck_pipeline.log"))
logger = logging.getLogger("scripts.run_scrape_deck_pipeline")


async def run_pipeline():
    TOTAL_PAGES = 17

    async with httpx.AsyncClient() as http_client, get_async_db_session() as db_session:
        extractor = FLDeckExtractor(deck_settings, http_client)
        transformer = DeckTransformer()

        for page_num in range(1, TOTAL_PAGES + 1):
            logger.info("Processing page number %s", page_num)

            page_of_decks = await extractor.get_page_of_decks(page_num)
            deck_ids = [int(deck.get("id")) for deck in page_of_decks]

            if not deck_ids:
                logger.warning("No decks found in page number %s", page_num)
                continue

            await aiometer.run_on_each(
                async_fn=functools.partial(
                    process_deck_wrapper,
                    extractor,
                    transformer,
                    db_session,
                ),
                args=deck_ids,
                max_at_once=deck_settings.AIOMETER_MAX_CONCURRENT,
                max_per_second=deck_settings.AIOMETER_MAX_PER_SECOND,
            )


async def process_deck_wrapper(
    extractor: FLDeckExtractor,
    transformer: DeckTransformer,
    db_session: AsyncSession,
    deck_id: int,
) -> None:
    try:
        await process_deck(extractor, transformer, db_session, deck_id)

    except RetryError:
        logger.error(
            "Deck %s failed after %s retries", deck_id, deck_settings.MAX_RETRIES
        )


@retry(
    stop=stop_after_attempt(deck_settings.MAX_RETRIES),
    wait=wait_exponential(
        multiplier=deck_settings.EXPONENTIAL_MULTIPLIER,
        min=deck_settings.EXPONENTIAL_MIN_WAIT,
        max=deck_settings.EXPONENTIAL_MAX_WAIT,
    ),
    retry=retry_if_exception_type((httpx.HTTPError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
)
async def process_deck(
    extractor: FLDeckExtractor,
    transformer: DeckTransformer,
    db_session: AsyncSession,
    deck_id: int,
) -> None:
    # Extract deck json
    deck_data = await extractor.get_deck(deck_id)

    # Transform deck json to dataframe
    transformed_df = transformer.transform_deck_data(deck_data, deck_id)

    # Load dataframe to database
    if transformed_df is not None and not transformed_df.empty:
        await load_deck_to_db(transformed_df, db_session)


async def load_deck_to_db(
    transformed_df: pd.DataFrame, db_session: AsyncSession
) -> None:
    # Sqlite doesn't handle pd.NA, change to None
    transformed_df = transformed_df.replace({pd.NA: None})

    # Create EventDeck object
    for row in transformed_df.itertuples():
        event_deck = EventDeck(
            published_at=row.published_at,
            deck_type=row.deck_type,
            deck_category=row.deck_category,
            deck_class=row.deck_class,
            card_name=row.card_name,
            card_amount=row.card_amount,
            deck_builder=row.deck_builder,
            event_name=row.event_name,
            event_placement=row.event_placement,
            url=row.url,
        )

        # Add object to sqlalchemy session
        db_session.add(event_deck)

    # Commit changes to database
    db_session.commit()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
