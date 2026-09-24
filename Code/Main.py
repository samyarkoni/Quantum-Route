import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Play-by-play data for 2022-2023, converted to pandas
games = pd.read_csv(SCRIPT_DIR.parent / "Data/games.csv")

print(games.head())



