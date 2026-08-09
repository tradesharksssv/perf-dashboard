"""
fetch_data.py
-------------
Non-interactive version of your kaggle-drive-connector notebook, safe to run
in GitHub Actions (headless, no browser popups, no files with secrets in them).

What changed vs. your notebook, and why:
  1. Drive download: uses the plain HTTP download function you already had
     (download_file_from_google_drive) instead of PyDrive2 + LocalWebserverAuth.
     LocalWebserverAuth() opens a browser window for you to click "Allow" --
     that cannot happen on a headless GitHub server, so it's removed.
     -> Your Drive file with the base CSV must be shared as "Anyone with the link".
  2. Chrome/driver: uses Selenium's built-in "Selenium Manager" (Selenium >=4.6),
     which auto-downloads a matching chromedriver. No pinned Chrome 94 / wheel
     files needed -- GitHub's runner just needs Chrome installed (handled in
     the workflow YAML via browser-actions/setup-chrome).
  3. Credentials: read from environment variables (set via GitHub Secrets),
     never from a JSON file. Nothing sensitive is written to disk or the repo.

Env vars required (set these as GitHub Secrets):
    ZERODHA_LOGIN_NAME, ZERODHA_PASSWORD, KITE_API_KEY, KITE_API_SECRET, KITE_TOTP_KEY
    BASE_CSV_DRIVE_FILE_ID   (the Google Drive file ID of your base PNL CSV)
"""

import os
import sys
import time
import json
import urllib.parse as urlparse

import pandas as pd
import pyotp
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from kiteconnect import KiteConnect

BASE_CSV_PATH = "daysPNL_dfUpdated_fixed.csv"


# ---------------------------------------------------------------------------
# 1. Non-interactive Google Drive download (no OAuth popup)
# ---------------------------------------------------------------------------

def download_file_from_google_drive(file_id: str, destination: str):
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def fetch_base_csv():
    file_id = os.environ["BASE_CSV_DRIVE_FILE_ID"]
    print(f"Downloading base CSV (Drive id {file_id})...")
    download_file_from_google_drive(file_id, BASE_CSV_PATH)
    df = pd.read_csv(BASE_CSV_PATH)
    if str(df.columns[0]).strip() in ("", "Unnamed: 0") or str(df.columns[0]).isdigit():
        df = df.iloc[:, 1:]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Loaded {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# 2. Headless Zerodha login -> Kite access token
# ---------------------------------------------------------------------------

def get_kite_session():
    api_key = os.environ["KITE_API_KEY"]
    api_secret = os.environ["KITE_API_SECRET"]
    login_name = os.environ["ZERODHA_LOGIN_NAME"]
    password = os.environ["ZERODHA_PASSWORD"]
    totp_key = os.environ["KITE_TOTP_KEY"]

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1280,900")

    # No explicit driver path: Selenium Manager resolves a matching driver itself.
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(login_url)
        time.sleep(3)

        WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.XPATH, "//div[@class='login-form']"))
        )
        driver.find_element(By.XPATH, "//input[@type='text']").send_keys(login_name)
        driver.find_element(By.XPATH, "//input[@type='password']").send_keys(password)
        time.sleep(2)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(5)

        WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.XPATH, "//div[@class='login-form']"))
        )
        time.sleep(2)
        totp = pyotp.TOTP(totp_key)
        driver.find_element(By.XPATH, "//input[@type='number']").send_keys(totp.now())
        time.sleep(2)
        try:
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
        except Exception:
            pass
        time.sleep(8)

        session_url = driver.current_url
        parsed = urlparse.urlparse(session_url)
        request_token = urlparse.parse_qs(parsed.query)["request_token"][0]
    finally:
        driver.quit()

    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print("Kite session established.")
    return kite


# ---------------------------------------------------------------------------
# 3. Mutual fund NAV history (no auth needed)
# ---------------------------------------------------------------------------

MF_TOKENS = {
    "Parag_Parikh_Flexi_Cap_Fund": "INF879O01027",
    "QUANTMUTUALFUND_MF": "INF966L01721",
    "SBIMutualFund_MF": "INF200K01UY4",
    "ICICIPrudentialMutualFund_MF": "INF109K018M4",
    "BirlaSunLifeMutualFund_MF": "INF209KB1O82",
    "INVESCOMUTUALFUND_MF": "INF205K01NG5",
    "MOTILALOSWAL_MF": "INF247L01445",
    "BANDHANMUTUALFUND_MF": "INF194KB1AL4",
    "EDELWEISSMUTUALFUND_MF": "INF843K01AO4",
    "HDFCMutualFund_MF": "INF179K01XQ0",
    "INVESCOMUTUALFUND_MidCap_MF": "INF205K01MV6",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_mf_history(isin_code: str) -> pd.DataFrame:
    index_response = requests.get("https://api.mfapi.in/mf", headers=HEADERS).json()
    scheme_code = None
    for scheme in index_response:
        if scheme.get("isinGrowth") == isin_code or scheme.get("isinDivReinvestment") == isin_code:
            scheme_code = scheme["schemeCode"]
            break
    if not scheme_code:
        raise ValueError(f"ISIN {isin_code} not found on MFapi index.")

    data_response = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", headers=HEADERS).json()
    df = pd.DataFrame(data_response["data"])
    df.columns = ["Date", "NAV"]
    df["NAV"] = pd.to_numeric(df["NAV"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_mf_returns() -> pd.DataFrame:
    mf_returns_df = None
    for col, isin in MF_TOKENS.items():
        print(f"  Fetching MF history: {col}")
        hist = get_mf_history(isin)
        hist["trade_date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
        hist[col] = hist["NAV"].pct_change() * 100
        hist = hist[["trade_date", col]]
        mf_returns_df = hist if mf_returns_df is None else mf_returns_df.merge(hist, on="trade_date", how="outer")
    return mf_returns_df[mf_returns_df["trade_date"] > "2024-01-01"]


# ---------------------------------------------------------------------------
# 4. Index historical data (needs Kite session)
# ---------------------------------------------------------------------------

INDEX_TOKENS = {
    "niftyReturns": 256265,
    "sensexReturns": 265,
    "niftyBankReturns": 260105,
    "niftyMidCapReturns": 266249,
    "niftySmallCapReturns": 267273,
}


def fetch_index_returns(kite, start_date="2024-01-01", end_date=None) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    returns_df = None
    for col, token in INDEX_TOKENS.items():
        print(f"  Fetching index history: {col}")
        hist = pd.DataFrame(kite.historical_data(token, start_date, end_date, "day"))
        hist["trade_date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None).dt.normalize()
        hist[col] = hist["close"].pct_change() * 100
        hist = hist[["trade_date", col]]
        returns_df = hist if returns_df is None else returns_df.merge(hist, on="trade_date", how="outer")
    return returns_df


# ---------------------------------------------------------------------------
# 5. Merge everything and save
# ---------------------------------------------------------------------------

def main():
    df = fetch_base_csv()

    kite = get_kite_session()
    index_returns = fetch_index_returns(kite)
    mf_returns = fetch_mf_returns()

    df["trade_date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(index_returns, on="trade_date", how="left")
    df = df.merge(mf_returns, on="trade_date", how="left")
    df = df.drop(columns=["trade_date"])

    df.to_csv("portfolio_data.csv", index=False)
    print(f"Saved merged dataset -> portfolio_data.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
