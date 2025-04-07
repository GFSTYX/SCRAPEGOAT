from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from gfwldata.config.settings import settings


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Synchronous engine and session
sync_engine = create_engine(settings.SYNC_DB_URL, pool_pre_ping=True)

SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session():
    """Synchronous database session context manager."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Async engine and session
async_engine = create_async_engine(
    settings.ASYNC_DB_URL,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


@asynccontextmanager
async def get_async_db_session():
    """Asynchronous database session context manager."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
