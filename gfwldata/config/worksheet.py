from pydantic import BaseModel, Field


class WorksheetSettings(BaseModel):
    FILEPATH: str = Field(description="Path of the GFWL xlsx file")
    SEASON: int = Field(description="The GFWL season of the file")
    WARS_HORIZONTAL: int = Field(
        description="The number of war matchups in a horizontal row in the matchups sheet"
    )
    WARS_VERTICAL: int = Field(
        description="The number of war matchups in a vertical row in the matchups sheet"
    )
    WEEK_SCAN_MAX_ROWS: int = Field(
        description="The maximum number of rows with data in the matchups sheet"
    )
    DECK_HISTORY_MAX_ROWS: int = Field(
        description="The maximum number of rows with data in the deck history sheet"
    )
    DECK_HISTORY_SHEET_NAME: str = Field(
        description="The sheet name that contains history of decks used in wars"
    )
    MATCHUPS_SHEET_NAME: str = Field(
        description="The sheet name of the weekly war matchups that contains the results and replays of each war"
    )


worksheet_settings: list[WorksheetSettings] = [
    WorksheetSettings(
        FILEPATH="gfwldata/data/sheets/gfwl-s1.xlsx",
        SEASON=1,
        WARS_HORIZONTAL=4,
        WARS_VERTICAL=2,
        WEEK_SCAN_MAX_ROWS=250,
        DECK_HISTORY_MAX_ROWS=200,
        DECK_HISTORY_SHEET_NAME="Deck Usage History",
        MATCHUPS_SHEET_NAME="Matchups + Film Archive",
    ),
    WorksheetSettings(
        FILEPATH="gfwldata/data/sheets/gfwl-s2.xlsx",
        SEASON=2,
        WARS_HORIZONTAL=4,
        WARS_VERTICAL=4,
        WEEK_SCAN_MAX_ROWS=400,
        DECK_HISTORY_MAX_ROWS=350,
        DECK_HISTORY_SHEET_NAME="Deck Usage History",
        MATCHUPS_SHEET_NAME="Matchups + Film Archive",
    ),
    WorksheetSettings(
        FILEPATH="gfwldata/data/sheets/gfwl-s3.xlsx",
        SEASON=3,
        WARS_HORIZONTAL=3,
        WARS_VERTICAL=4,
        WEEK_SCAN_MAX_ROWS=350,
        DECK_HISTORY_MAX_ROWS=300,
        DECK_HISTORY_SHEET_NAME="Deck Usage",
        MATCHUPS_SHEET_NAME="Matchups + Film",
    ),
    WorksheetSettings(
        FILEPATH="gfwldata/data/sheets/gfwl-s4.xlsx",
        SEASON=4,
        WARS_HORIZONTAL=3,
        WARS_VERTICAL=4,
        WEEK_SCAN_MAX_ROWS=350,
        DECK_HISTORY_MAX_ROWS=300,
        DECK_HISTORY_SHEET_NAME="Deck Usage",
        MATCHUPS_SHEET_NAME="Matchups + Film",
    ),
    WorksheetSettings(
        FILEPATH="gfwldata/data/sheets/gfwl-s5.xlsx",
        SEASON=5,
        WARS_HORIZONTAL=3,
        WARS_VERTICAL=4,
        WEEK_SCAN_MAX_ROWS=350,
        DECK_HISTORY_MAX_ROWS=300,
        DECK_HISTORY_SHEET_NAME="Deck Usage",
        MATCHUPS_SHEET_NAME="Matchups + Film",
    ),
]
