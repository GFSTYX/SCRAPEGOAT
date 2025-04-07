# notebooks

This folder contains Jupyter Notebooks for data cleaning, analysis, and modeling, using data from the [scripts](/scripts) pipelines. Each directory contains a single notebook and a rendered README generated using `nbconvert`.

## Directory Structure

- [clean_tables](./clean_tables): Notebook for cleaning database tables (e.g., removing duplicated `replay_url` entries). See `README.md` for details.
- [deck_model](./deck_model): Notebook for building a deck classification model to predict deck classes (or types) using data from FormatLibrary. See `README.md` for details.
- [deck_model_demo](./deck_model_demo): Notebook that demos the model from [deck_model](./deck_model) to better understand it's performance.
- [gfwl_analysis](./gfwl_analysis): Notebook that shows exploratory data analysis on goat format war league data.
