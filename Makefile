# Force Git Bash on Windows 
SHELL := C:/Program Files/Git/bin/bash.exe

.PHONY: clean lint help db-init db-open db-delete db-reset convert-md

# Show help by default
.DEFAULT_GOAL := help

# Linting
lint:
	uv run ruff check .
	uv run ruff format .

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Database commands
db-init:
	uv run py -m scripts.init_db

db-open:
	sqlite3 gfwldata/data/gfwldata.db

db-delete:
	rm -f gfwldata/data/gfwldata.db

db-reset: db-delete db-init

# Notebook commands
convert-nb:
	uv run jupyter nbconvert --to markdown "$(file)" --output "README.md"

# Help command
help:
	@echo "Available commands:"
	@echo "  lint         : Run code formatters and linters"
	@echo "  clean        : Remove Python cache files and build artifacts"
	@echo "  db-init      : Initialize database tables"
	@echo "  db-open      : Open SQLite database session"
	@echo "  db-delete    : Delete the SQLite database file"
	@echo "  db-reset     : Delete and reinitialize the database"
	@echo "  convert-nb   : Convert Jupyter notebook to README.md (use with file=path/to/notebook.ipynb)"
