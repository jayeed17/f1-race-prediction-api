import os
import time
from typing import List, Dict, Any

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"

MAX_RETRIES = 5


def get_json(url: str) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** (attempt + 1)))
            print(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Exceeded max retries fetching {url}")


def get_schedule(season: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/{season}.json"
    data = get_json(url)
    return data["MRData"]["RaceTable"]["Races"]


def get_race_results(season: int, round_num: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/{season}/{round_num}/results.json?limit=100"
    data = get_json(url)
    races = data["MRData"]["RaceTable"]["Races"]
    return races[0]["Results"] if races else []


def get_qualifying_results(season: int, round_num: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/{season}/{round_num}/qualifying.json?limit=100"
    data = get_json(url)
    races = data["MRData"]["RaceTable"]["Races"]
    return races[0]["QualifyingResults"] if races else []


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_season_dataset(season: int) -> pd.DataFrame:
    races = get_schedule(season)
    rows = []

    for race in races:
        round_num = int(race["round"])
        race_name = race["raceName"]
        circuit_name = race["Circuit"]["circuitName"]

        print(f"Processing {season} Round {round_num}: {race_name}")

        results = get_race_results(season, round_num)
        qualifying = get_qualifying_results(season, round_num)

        qualifying_map = {}
        for q in qualifying:
            driver_id = q["Driver"]["driverId"]
            qualifying_map[driver_id] = safe_int(q.get("position"))

        for result in results:
            driver = result["Driver"]
            constructor = result["Constructor"]

            driver_id = driver["driverId"]
            driver_name = f'{driver["givenName"]} {driver["familyName"]}'
            team_name = constructor["name"]

            grid_position_int = safe_int(result.get("grid"))
            finishing_position_int = safe_int(result.get("position"))
            qualifying_position_int = qualifying_map.get(driver_id)

            podium_label = (
                1 if finishing_position_int is not None and finishing_position_int <= 3 else 0
            )

            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race_name,
                    "circuit_name": circuit_name,
                    "driver": driver_name,
                    "team": team_name,
                    "grid_position": grid_position_int,
                    "qualifying_position": qualifying_position_int,
                    "finishing_position": finishing_position_int,
                    "podium_label": podium_label,
                }
            )

        time.sleep(0.2)

    return pd.DataFrame(rows)


def build_dataset(start_season: int, end_season: int) -> pd.DataFrame:
    all_dfs = []

    for season in range(start_season, end_season + 1):
        season_df = build_season_dataset(season)
        all_dfs.append(season_df)

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    df = build_dataset(2023, 2024)

    df = df.drop_duplicates()

    df = df[
        [
            "season",
            "round",
            "race_name",
            "circuit_name",
            "driver",
            "team",
            "grid_position",
            "qualifying_position",
            "finishing_position",
            "podium_label",
        ]
    ]

    output_path = "data/raw/f1_race_prediction_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nDataset saved successfully.")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print("\nPreview:")
    print(df.head())