"""Collect a reproducible sample of public Kuku FM Google Play reviews."""
from pathlib import Path
import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = "com.vlv.aravali"  # Verified Kuku FM Google Play package ID
LANGUAGE = "en"
COUNTRY = "in"
SAMPLE_SIZE = 200
OUTPUT = Path(__file__).parent / "kuku_fm_reviews_en_in.csv"

result, _ = reviews(
    APP_ID,
    lang=LANGUAGE,
    country=COUNTRY,
    sort=Sort.NEWEST,
    count=SAMPLE_SIZE,
)

# Keep fields useful for analysis; omit reviewer names, IDs, and profile images.
columns = ["content", "score", "at", "appVersion", "thumbsUpCount", "replyContent"]
df = pd.DataFrame(result)
df = df[[column for column in columns if column in df.columns]]
if "content" in df.columns:
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["content"])
if "at" in df.columns:
    df["at"] = pd.to_datetime(df["at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} reviews to {OUTPUT}")
print("Columns:", ", ".join(df.columns))
print("Sample rows:")
print(df.head(3).to_string(index=False))
