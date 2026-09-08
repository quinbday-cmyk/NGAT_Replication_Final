import os
import yfinance as yf
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

TICKER_FOLDER = r"C:\Users\quinb\Desktop\UW CFRM\522\Research Paper 1\522 Replication Code\NGAT\data\raw\SPNews\price\raw"       # folder containing aapl.csv, aa.csv, etc
OUTPUT_FOLDER = r"C:\Users\quinb\Desktop\UW CFRM\522\Research Paper 1\522 Replication Code\NGAT\data\raw\New Data Generation"

START_DATE = "2012-01-01"
END_DATE = "2026-02-07"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# EXTRACT TICKERS FROM FILENAMES
# -----------------------------

tickers = []

for file in os.listdir(TICKER_FOLDER):
    if file.endswith(".csv"):
        ticker = os.path.splitext(file)[0].upper()
        tickers.append(ticker)

tickers = list(set(tickers))   # remove duplicates

print(f"Found {len(tickers)} tickers")

# -----------------------------
# DOWNLOAD & SAVE
# -----------------------------

for ticker in tickers:
    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False
    )

    if df.empty:
        print(f"⚠ No data for {ticker}")
        continue

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.reset_index(inplace=True)

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    df.to_csv(
        os.path.join(OUTPUT_FOLDER, f"{ticker}.csv"),
        index=False
    )

print("\nAll done.")
