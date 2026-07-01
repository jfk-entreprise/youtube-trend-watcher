# YouTube Trend Watcher

Monitors YouTube trending videos and surfaces insights via the YouTube Data API v3.

## Requirements

- Python 3.10+
- A [YouTube Data API v3](https://console.cloud.google.com/apis/library/youtube.googleapis.com) key

## Setup

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# Then edit .env and fill in your API key
```

## Project structure

```
youtube-trend-watcher/
├── .venv/               # Virtual environment (not committed)
├── .env                 # Local secrets (not committed)
├── .env.example         # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

_To be documented as features are implemented._

## License

MIT
