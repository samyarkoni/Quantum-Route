import nfl_data_py as nfl
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load data
df = read.csv('nfl_data.csv')

pbp = nfl.import_pbp_data([2022, 2023])

# Let's see what play types we have
print("Play Types:")
print(pbp['play_type'].value_counts())
print("\n")

# Check what teams are in the data
print("Home Teams Sample:")
print(pbp['home_team'].value_counts().head())
print("\n")

# Look at one game
sample_game = pbp[pbp['game_id'] == pbp['game_id'].iloc[0]]
print(f"Sample game has {len(sample_game)} plays")
print(f"Teams: {sample_game['home_team'].iloc[0]} vs {sample_game['away_team'].iloc[0]}")

