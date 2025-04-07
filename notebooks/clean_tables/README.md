# Clean database tables

Going to remove any duplicated replay_url's from league_matches and jobs tables.

## Table of Contents

1. [Clean tables](#clean-tables)
    - [Imports](#imports)
    - [Configurations](#configurations)
    - [Query all data from league_matches](#query-all-data-from-league_matches)
    - [Find all duplicated replay_urls](#find-all-duplicated-replay_urls)
    - [Go through each to find incorrect match data](#go-through-each-to-find-incorrect-match-data)
    - [Collect league_matches id's to remove](#collect-league_matches-ids-to-remove)
1. [Remove duplicated ids from database](#remove-duplicated-ids-from-database)
    - [Imports](#imports-1)
    - [Remove ids from league_matches and jobs tables](#remove-ids-from-league_matches-and-jobs-tables)

## Clean tables

### Imports


```python
import polars as pl

from gfwldata.utils.db import sync_engine
```

### Configurations


```python
# Don't cut off column length configuration
pl.Config(fmt_str_lengths=100)
```




    <polars.config.Config at 0x21d4a705040>



### Query all data from league_matches


```python
raw_tbl = pl.read_database(
    query="select * from league_matches",
    connection=sync_engine
)
```

### Find all duplicated replay_urls


```python
with pl.Config(fmt_str_lengths=100):
    replay_url_duplicates = (
        raw_tbl
        .filter(pl.col("replay_url").is_not_null())
        .group_by("replay_url")
        .len()
        .filter(pl.col("len") > 1)
    )
    
    print(replay_url_duplicates)
```

    shape: (4, 2)
    ┌────────────────────────────────────────────┬─────┐
    │ replay_url                                 ┆ len │
    │ ---                                        ┆ --- │
    │ str                                        ┆ u32 │
    ╞════════════════════════════════════════════╪═════╡
    │ https://duelingbook.com/replay?id=68321903 ┆ 2   │
    │ https://duelingbook.com/replay?id=67849343 ┆ 2   │
    │ https://duelingbook.com/replay?id=67778738 ┆ 2   │
    │ https://duelingbook.com/replay?id=55940660 ┆ 2   │
    └────────────────────────────────────────────┴─────┘
    

### Go through each to find incorrect match data


```python
(
    raw_tbl
    .filter(pl.col("replay_url").is_in(replay_url_duplicates["replay_url"].to_list()))
    .select(pl.exclude(["replay_id", "created_at", "updated_at"]))
    .sort(["replay_url"])
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (8, 11)</small><table border="1" class="dataframe"><thead><tr><th>season</th><th>week</th><th>team1</th><th>team2</th><th>team1_player</th><th>team2_player</th><th>team1_player_deck_type</th><th>team2_player_deck_type</th><th>match_score</th><th>replay_url</th><th>id</th></tr><tr><td>i64</td><td>i64</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>4</td><td>3</td><td>&quot;The Worst Generation&quot;</td><td>&quot;Rehab&quot;</td><td>&quot;don&#x27;t copy&quot;</td><td>&quot;insids&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;2-0&quot;</td><td>&quot;https://duelingbook.com/replay?id=55940660&quot;</td><td>&quot;bfda413aac224bcaabd7a9ecb8fb7d3f&quot;</td></tr><tr><td>4</td><td>3</td><td>&quot;Masters of Oz&quot;</td><td>&quot;Purple Haze&quot;</td><td>&quot;jase&quot;</td><td>&quot;raysaber&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;1-2&quot;</td><td>&quot;https://duelingbook.com/replay?id=55940660&quot;</td><td>&quot;d6d4c00c5f06465b9d0ce393e8992ca3&quot;</td></tr><tr><td>6</td><td>4</td><td>&quot;team_bomb_squad&quot;</td><td>&quot;team_batl_oxen&quot;</td><td>&quot;The dark knight&quot;</td><td>&quot;psyt&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;Cyber-Stein OTK&quot;</td><td>&quot;2-1&quot;</td><td>&quot;https://duelingbook.com/replay?id=67778738&quot;</td><td>&quot;63e653b80fee48728ba4c0a8debf5834&quot;</td></tr><tr><td>6</td><td>4</td><td>&quot;team_batl_oxen&quot;</td><td>&quot;team_bomb_squad&quot;</td><td>&quot;psyt&quot;</td><td>&quot;The dark knight&quot;</td><td>&quot;Cyber-Stein OTK&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;2-0&quot;</td><td>&quot;https://duelingbook.com/replay?id=67778738&quot;</td><td>&quot;e230d51d0a604b21939244d47a4ee3df&quot;</td></tr><tr><td>6</td><td>4</td><td>&quot;team_black_luster_serbs&quot;</td><td>&quot;team_4hunnids&quot;</td><td>&quot;popz&quot;</td><td>&quot;Lukaz&quot;</td><td>&quot;Reasoning Gate Turbo&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;2-1&quot;</td><td>&quot;https://duelingbook.com/replay?id=67849343&quot;</td><td>&quot;d3054c2c57d048939849a2e85e1c7ad2&quot;</td></tr><tr><td>6</td><td>4</td><td>&quot;team_black_luster_serbs&quot;</td><td>&quot;team_4hunnids&quot;</td><td>&quot;popz&quot;</td><td>&quot;SolMasterMatt&quot;</td><td>&quot;Reasoning Gate Turbo&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;2-1&quot;</td><td>&quot;https://duelingbook.com/replay?id=67849343&quot;</td><td>&quot;f596b7ca06c0460591b2e9ee2d73e30d&quot;</td></tr><tr><td>6</td><td>6</td><td>&quot;team_the_bones_generation&quot;</td><td>&quot;team_carbombnara&quot;</td><td>&quot;Shifty&quot;</td><td>&quot;CCaliendo&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;Chaos Control&quot;</td><td>&quot;2-1&quot;</td><td>&quot;https://duelingbook.com/replay?id=68321903&quot;</td><td>&quot;19226427495a400a99792badbb63fe13&quot;</td></tr><tr><td>6</td><td>6</td><td>&quot;team_the_bones_generation&quot;</td><td>&quot;team_carbombnara&quot;</td><td>&quot;Shifty&quot;</td><td>&quot;marcor96&quot;</td><td>&quot;Chaos Turbo&quot;</td><td>&quot;Chaos Control&quot;</td><td>&quot;2-1&quot;</td><td>&quot;https://duelingbook.com/replay?id=68321903&quot;</td><td>&quot;5c25d683cdbd449eaaa9e1ea2e9e155a&quot;</td></tr></tbody></table></div>



### Collect league_matches id's to remove


```python
incorrect_matches_id = [
    "bfda413aac224bcaabd7a9ecb8fb7d3f",
    "e230d51d0a604b21939244d47a4ee3df",
    "d3054c2c57d048939849a2e85e1c7ad2",
    "19226427495a400a99792badbb63fe13"
]
```

## Remove duplicated ids from database

### Imports


```python
from uuid import UUID

from sqlalchemy import delete

from gfwldata.utils.db import get_db_session
from gfwldata.utils.models import Job, LeagueMatch
```

### Remove ids from league_matches and jobs tables


```python
# Convert string of ids to uuid
league_ids_to_delete = [UUID(id) for id in incorrect_matches_id]

# Delete duplicated ids from league_matches table
with get_db_session() as session:
    stmt = delete(LeagueMatch).where(LeagueMatch.id.in_(league_ids_to_delete))
    result = session.execute(stmt)
    
    session.commit()
    
# Delete duplicated ids from jobs table
with get_db_session() as session:
    stmt = delete(Job).where(Job.league_match_id.in_(league_ids_to_delete))
    result = session.execute(stmt)
    
    session.commit()
```
