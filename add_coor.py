import pandas as pd
import requests
import time
import json
import os

CACHE_FILE = "geocode_cache.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "NLPProject/1.0 (gaetan.fichetdc@gmail.com)"}

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def geocode_location(loc, session, cache):
    if loc in cache:
        return cache[loc]
    try:
        r = session.get(NOMINATIM_URL, params={"q": loc, "format": "json", "limit": 1}, headers=HEADERS, timeout=10)
        if r.status_code == 200 and r.json():
            data = r.json()[0]
            result = [float(data["lat"]), float(data["lon"])]
            cache[loc] = result
            return result
    except Exception as e:
        print(f"  Failed: {loc} ({e})")
    cache[loc] = [None, None]
    return [None, None]

def main():
    print("Reading dataset...")
    df = pd.read_csv("ai_jobs_global.csv")

    if "city" not in df.columns:
        print("Error: No 'city' column found!")
        return

    country_col = ", " + df["country"].astype(str) if "country" in df.columns else ""
    df["full_location"] = df["city"].astype(str) + country_col

    unique_locations = df["full_location"].dropna().unique()
    cache = load_cache()
    to_geocode = [loc for loc in unique_locations if loc not in cache]
    print(f"Found {len(unique_locations)} unique locations, {len(to_geocode)} need geocoding.")

    with requests.Session() as session:
        for i, loc in enumerate(to_geocode):
            geocode_location(loc, session, cache)
            if (i + 1) % 50 == 0:
                save_cache(cache)
                print(f"  Progress: {i + 1}/{len(to_geocode)}")
            time.sleep(1)

    save_cache(cache)

    df["lat"] = df["full_location"].map(lambda x: cache.get(x, [None, None])[0])
    df["lon"] = df["full_location"].map(lambda x: cache.get(x, [None, None])[1])

    output_filename = "ai_jobs_global_geocoded.csv"
    df.to_csv(output_filename, index=False)
    geocoded = df["lat"].notna().sum()
    print(f"Done! {geocoded}/{len(df)} rows geocoded. Saved to '{output_filename}'")

if __name__ == "__main__":
    main()