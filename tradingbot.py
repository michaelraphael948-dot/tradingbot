#!/usr/bin/env python3
"""
tradingbot.py  -- Super-long giant fixed file (based on last working version)
Features:
- Bybit v5 klines (live candlesticks)
- Plotly (light theme) PNG export via kaleido (Chrome required)
- Mathematically-detected Supply & Demand zones (volume spike + pivot + confirming move)
- Shaded rectangles + small markers labeling S/D zones
- normalize_interval() supports many user inputs for /setinterval and /chart
- All requested commands: /start, /help, /status, /list, /chart, /analyze,
  /price, /setinterval, /sethour, /setletter, /setletters, /setsuffix,
  /setmin, /setmax, /settestpattern, /pause, /resume, /test, /export, /import,
  /allletters, /diag
- Per-chat persistent JSON config (./data/chat_configs.json)
- Scheduler daily auto-send at configured UTC hour
- Supervisor loop auto-restarts the bot on crash
- Defensive HTTP with retries, caching, fallback intervals
- Logging to console + file
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
import re
from io import BytesIO
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Ensure kaleido/static image support installed in your venv:
# pip install plotly[kaleido] kaleido
# And ensure Google Chrome/Chromium is installed (kaleido needs chrome)
# If missing, run: plotly_get_chrome

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -------------------------
# Load environment
# -------------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "").strip()  # optional
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()  # optional

# Interactive prompt for TELEGRAM_TOKEN if missing (convenience)
_token_pattern = re.compile(r"^\d{6,}:[A-Za-z0-9_\-]{20,}$")
if not TELEGRAM_TOKEN:
    print("\nNo TELEGRAM_TOKEN found in .env. Paste it now (or Ctrl-C to abort):")
    try:
        raw = input("TELEGRAM_TOKEN: ").strip()
        if _token_pattern.match(raw):
            TELEGRAM_TOKEN = raw
        else:
            print("Token format looks invalid. Exiting.")
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)

# -------------------------
# Config & paths
# -------------------------
DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
CHAT_CONFIG_FILE = os.path.join(DATA_DIR, "chat_configs.json")
LOG_FILE = os.path.join(DATA_DIR, "tradingbot_superlong.log")

BYBIT_BASE = "https://api.bybit.com"

# canonical intervals the bot understands
VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"}
# map canonical to Bybit interval codes used earlier (we'll support many)
MAP_INTERVAL = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M"
}

DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT = 200
DEFAULT_SUFFIX = "USDT"
DEFAULT_HOUR_UTC = 9
DEFAULT_TEST_PATTERN = [20, 20, 20]
MAX_CHARTS_SAFE = 20
CACHE_TTL = 3600  # seconds
HTTP_TIMEOUT = 20  # seconds

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tradingBot_superlong: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("tradingBot_superlong")

logger.info("DEBUG: TELEGRAM_TOKEN=%s", (TELEGRAM_TOKEN[:10] + "...") if TELEGRAM_TOKEN else None)
logger.info("DEBUG: BYBIT_API_KEY=%s", (BYBIT_API_KEY[:6] + "...") if BYBIT_API_KEY else None)

# -------------------------
# Globals
# -------------------------
_symbol_cache: Dict[str, Tuple[float, Any]] = {}
_chat_configs: Dict[str, Any] = {}

# -------------------------
# Persistence helpers
# -------------------------
def load_json(path: str) -> Any:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load JSON: %s", path)
        return None

def save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        logger.exception("Failed to save JSON: %s", path)

_loaded_cfg = load_json(CHAT_CONFIG_FILE)
if isinstance(_loaded_cfg, dict):
    _chat_configs = _loaded_cfg
else:
    _chat_configs = {}

def default_chat_config() -> Dict[str, Any]:
    return {
        "interval": DEFAULT_INTERVAL,
        "hour": DEFAULT_HOUR_UTC,
        "suffix": DEFAULT_SUFFIX,
        "min_candles": 50,
        "max_charts": 6,
        "paused": False,
        "custom_letters": None,
        "forced_letter": None,
        "last_run": None,
        "test_pattern": DEFAULT_TEST_PATTERN,
        "require_binance_match": False
    }

def ensure_chat_config(chat_id: int) -> Dict[str, Any]:
    key = str(chat_id)
    if key not in _chat_configs:
        _chat_configs[key] = default_chat_config()
        save_json(CHAT_CONFIG_FILE, _chat_configs)
    return _chat_configs[key]

def save_chat_configs():
    save_json(CHAT_CONFIG_FILE, _chat_configs)

# -------------------------
# HTTP helper with retries
# -------------------------
def http_get_json(url: str, params: dict = None, max_retries: int = 4, backoff: float = 2.0, timeout: int = HTTP_TIMEOUT):
    params = params or {}
    attempt = 0
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            r = requests.get(url, params=params, timeout=timeout)
            text = r.text[:2000] if r.text else ""
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return text
            else:
                last_exc = RuntimeError(f"HTTP {r.status_code}: {text}")
                logger.warning("GET %s returned %s (attempt %d/%d)", url, r.status_code, attempt, max_retries)
                if r.status_code == 429:
                    time.sleep(5 * attempt)
                else:
                    time.sleep(backoff ** (attempt - 1))
        except Exception as e:
            last_exc = e
            logger.warning("GET %s failed on attempt %d/%d: %s", url, attempt, max_retries, e)
            time.sleep(backoff ** (attempt - 1))
    raise RuntimeError(f"Failed GET {url} after {max_retries} attempts: {last_exc}")

# -------------------------
# Bybit symbol list (cached)
# -------------------------
def fetch_bybit_symbols(force: bool = False) -> List[str]:
    key = "bybit_symbols"
    now = time.time()
    if not force and key in _symbol_cache:
        ts, val = _symbol_cache[key]
        if now - ts < CACHE_TTL:
            return val
    url = BYBIT_BASE + "/v5/market/instruments-info"
    params = {"category": "linear"}
    try:
        data = http_get_json(url, params=params)
        syms = []
        if isinstance(data, dict):
            for item in data.get("result", {}).get("list", []):
                sym = item.get("symbol")
                if sym:
                    syms.append(sym)
        _symbol_cache[key] = (now, syms)
        try:
            with open(os.path.join(DATA_DIR, "bybit_symbols_cache.json"), "w", encoding="utf-8") as f:
                json.dump(syms, f, indent=2)
        except Exception:
            pass
        return syms
    except Exception:
        logger.exception("Failed to fetch bybit symbols")
        if key in _symbol_cache:
            return _symbol_cache[key][1]
        try:
            cachepath = os.path.join(DATA_DIR, "bybit_symbols_cache.json")
            if os.path.exists(cachepath):
                with open(cachepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

# -------------------------
# Defensive Bybit kline fetch
# -------------------------
def get_bybit_klines(symbol: str, interval: str = "1h", limit: int = DEFAULT_LIMIT, log_raw: bool = False) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError("Invalid interval")
    bybit_interval = MAP_INTERVAL.get(interval, MAP_INTERVAL.get(DEFAULT_INTERVAL))
    url = BYBIT_BASE + "/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": str(limit)}
    try:
        data = http_get_json(url, params=params, max_retries=4)
    except Exception as e:
        logger.warning("Kline fetch HTTP failed for %s %s: %s", symbol, interval, e)
        return pd.DataFrame()

    if log_raw:
        try:
            logger.debug("RAW kline for %s: %s", symbol, str(data)[:2000])
        except Exception:
            pass

    rows = []
    if isinstance(data, dict):
        rows = data.get("result", {}).get("list") or []
    elif isinstance(data, list):
        rows = data
    else:
        logger.warning("Unexpected kline payload type for %s: %s", symbol, type(data))
        rows = []

    if not rows:
        # fallback for sparse markets
        if interval in ("1h", "4h", "1d"):
            logger.info("No rows for %s at %s; trying fallback 15m", symbol, interval)
            time.sleep(0.3)
            return get_bybit_klines(symbol, interval="15m", limit=limit, log_raw=log_raw)
        return pd.DataFrame()

    try:
        df = pd.DataFrame(rows)
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        else:
            logger.warning("Kline rows for %s have unexpected shape: %s", symbol, df.shape)
            return pd.DataFrame()

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp"].max() > 1e12:
            df["timestamp"] = (df["timestamp"] / 1000).astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)
        df = df.sort_index()
        return df
    except Exception:
        logger.exception("Failed to parse kline rows for %s", symbol)
        return pd.DataFrame()

# -------------------------
# Mathematical Supply & Demand detection
# -------------------------
def detect_supply_demand_zones(
    df: pd.DataFrame,
    lookback: int = 50,
    vol_multiplier: float = 1.5,
    price_window: int = 3,
    subsequent_move_pct: float = 0.03
) -> Tuple[List[Tuple[float, float, int, int, float]], List[Tuple[float, float, int, int, float]]]:
    sells = []
    buys = []
    try:
        if df.empty or len(df) < (lookback + price_window + 2):
            return sells, buys

        vol_avg = df["volume"].rolling(window=lookback, min_periods=1).mean()
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        for i in range(price_window, n - price_window - 1):
            # pivot high
            window_high = highs[i - price_window:i + price_window + 1]
            if highs[i] == window_high.max():
                if df["volume"].iat[i] > vol_multiplier * vol_avg.iat[i]:
                    look_ahead = min(n - 1, i + price_window)
                    subsequent_min = closes[i + 1: look_ahead + 1].min() if i + 1 <= look_ahead else closes[i + 1]
                    move_pct = (closes[i] - subsequent_min) / closes[i] if closes[i] else 0
                    if move_pct >= subsequent_move_pct:
                        ph = highs[i]
                        pl = closes[i] * 0.995
                        si = max(0, i - lookback)
                        ei = min(n - 1, i + price_window)
                        score = (df["volume"].iat[i] / (vol_avg.iat[i] + 1e-9)) * move_pct
                        sells.append((pl, ph, si, ei, float(score)))
            # pivot low
            window_low = lows[i - price_window:i + price_window + 1]
            if lows[i] == window_low.min():
                if df["volume"].iat[i] > vol_multiplier * vol_avg.iat[i]:
                    look_ahead = min(n - 1, i + price_window)
                    subsequent_max = closes[i + 1: look_ahead + 1].max() if i + 1 <= look_ahead else closes[i + 1]
                    move_pct = (subsequent_max - closes[i]) / closes[i] if closes[i] else 0
                    if move_pct >= subsequent_move_pct:
                        pl = lows[i]
                        ph = closes[i] * 1.005
                        si = max(0, i - lookback)
                        ei = min(n - 1, i + price_window)
                        score = (df["volume"].iat[i] / (vol_avg.iat[i] + 1e-9)) * move_pct
                        buys.append((pl, ph, si, ei, float(score)))
    except Exception:
        logger.exception("Zone detection failure")
    sells.sort(key=lambda x: x[4], reverse=True)
    buys.sort(key=lambda x: x[4], reverse=True)
    return sells, buys

# -------------------------
# Plotly light-mode chart with zones and markers
# -------------------------
def plotly_candles_with_zones_light(df: pd.DataFrame, symbol: str, interval: str,
                                    lookback: int = 50, vol_multiplier: float = 1.8) -> BytesIO:
    if df.empty:
        raise ValueError("Empty dataframe")

    df_plot = df.copy()
    if df_plot.index.tz is not None:
        df_plot.index = df_plot.index.tz_convert(None)

    sells, buys = detect_supply_demand_zones(df_plot, lookback=lookback, vol_multiplier=vol_multiplier)

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        increasing_line_color='#008000',
        decreasing_line_color='#D62728',
        showlegend=False
    ))

    # volume
    volumes = df_plot['volume']
    max_vol = volumes.max() if not volumes.empty else 1
    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=volumes,
        marker_color='rgba(120,120,120,0.4)',
        yaxis='y2',
        showlegend=False
    ))

    dates = list(df_plot.index.to_pydatetime())
    n = len(df_plot)

    # sells
    for idx, (pl, ph, si, ei, score) in enumerate(sells):
        try:
            x0 = dates[si]
            x1 = dates[ei]
            fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=pl, y1=ph,
                          fillcolor="rgba(255,100,100,0.18)", line_width=0)
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=pl, y1=pl,
                          line=dict(color="rgba(255,80,80,0.6)", dash="dash", width=1))
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=ph, y1=ph,
                          line=dict(color="rgba(255,80,80,0.6)", dash="dash", width=1))
            center_x = dates[(si + ei) // 2] if (si + ei) // 2 < n else dates[ei]
            center_y = ph
            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                text=[f"S{idx+1}"],
                textposition="top center",
                showlegend=False
            ))
        except Exception:
            continue

    # buys
    for idx, (pl, ph, si, ei, score) in enumerate(buys):
        try:
            x0 = dates[si]
            x1 = dates[ei]
            fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=pl, y1=ph,
                          fillcolor="rgba(100,200,120,0.12)", line_width=0)
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=pl, y1=pl,
                          line=dict(color="rgba(0,150,0,0.4)", dash="dash", width=1))
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=ph, y1=ph,
                          line=dict(color="rgba(0,150,0,0.4)", dash="dash", width=1))
            center_x = dates[(si + ei) // 2] if (si + ei) // 2 < n else dates[ei]
            center_y = pl
            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                text=[f"D{idx+1}"],
                textposition="bottom center",
                showlegend=False
            ))
        except Exception:
            continue

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='rgba(250,250,250,1)',
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(rangeslider=dict(visible=False), showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(side='right', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
        yaxis2=dict(overlaying='y', side='left', position=0.06, showgrid=False, title='Volume', range=[0, max_vol*5 if max_vol else 1]),
        title=f"{symbol} — {interval} — Supply/Demand zones"
    )

    fig.update_xaxes(range=[df_plot.index[0], df_plot.index[-1]])

    try:
        img_bytes = pio.to_image(fig, format="png", width=1400, height=800, scale=1)
        buf = BytesIO(img_bytes)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.exception("Plotly to_image failed: %s", e)
        try:
            tmpfn = os.path.join(DATA_DIR, f"{symbol}_{interval}_tmp.png")
            fig.write_image(tmpfn, width=1400, height=800)
            with open(tmpfn, "rb") as f:
                b = BytesIO(f.read())
            try:
                os.remove(tmpfn)
            except Exception:
                pass
            b.seek(0)
            return b
        except Exception:
            logger.exception("Plotly fallback also failed")
            raise

# -------------------------
# Sync wrapper for plotting used by run_in_executor
# -------------------------
def _sync_plot_bytes_safe(symbol: str, interval: str, lookback: int = 50, vol_multiplier: float = 1.8) -> BytesIO:
    bybit_list = fetch_bybit_symbols()
    if bybit_list and symbol not in bybit_list:
        raise RuntimeError(f"Symbol {symbol} not found in Bybit instruments-info")

    df = get_bybit_klines(symbol, interval=interval, limit=DEFAULT_LIMIT, log_raw=True)
    if df.empty and interval in ("1h", "4h", "1d"):
        df = get_bybit_klines(symbol, interval="15m", limit=DEFAULT_LIMIT, log_raw=True)
    if df.empty:
        raise RuntimeError(f"No kline data for {symbol} at {interval}")
    if len(df) < 12:
        raise RuntimeError(f"Insufficient candles for {symbol} ({len(df)}). Increase limit or use different interval.")

    buf = plotly_candles_with_zones_light(df, symbol, interval, lookback=lookback, vol_multiplier=vol_multiplier)
    return buf

# -------------------------
# normalize_interval utility
# -------------------------
def normalize_interval(raw: str) -> Optional[str]:
    """
    Normalize many user input variants into canonical intervals from VALID_INTERVALS.
    Accepts digits (15 -> 15m), '60' -> 1h, '1H', '60m', '1d', 'D', 'week', '1M', etc.
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    # direct match
    smap = {
        "1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "240": "4h",
        "1440": "1d", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
        "1d": "1d", "d": "1d", "day": "1d", "1w": "1w", "w": "1w", "1month": "1M", "1mo":"1M", "1mth":"1M"
    }
    if s in smap:
        val = smap[s]
        return val if val in VALID_INTERVALS else None
    # numeric only
    if s.isdigit():
        n = int(s)
        if n == 1:
            return "1m"
        if n in (5, 15, 30):
            return f"{n}m"
        if n == 60:
            return "1h"
        if n == 120:
            return "2h"
        if n == 240:
            return "4h"
        if n == 360:
            return "6h"
        if n == 720:
            return "12h"
        if n == 1440:
            return "1d"
        return None
    # capture patterns
    m = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hour|hours|d|day|days|w|week|mo|month|months)$', s)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if unit.startswith('m') and unit != 'mo':
            if val in (1,5,15,30):
                return f"{val}m"
            if val == 60:
                return "1h"
            if val == 240:
                return "4h"
            return None
        if unit.startswith('h'):
            if val == 1:
                return "1h"
            if val == 2:
                return "2h"
            if val == 4:
                return "4h"
            if val == 6:
                return "6h"
            if val == 12:
                return "12h"
            return None
        if unit.startswith('d'):
            if val == 1:
                return "1d"
            return None
        if unit.startswith('w'):
            if val == 1:
                return "1w"
            return None
        if unit in ("mo","month","months"):
            if val == 1:
                return "1M"
            return None
    # synonyms
    syn = {
        "60m": "1h", "60min":"1h", "1hour":"1h", "onehour":"1h", "hour":"1h",
        "day":"1d", "daily":"1d", "week":"1w", "monthly":"1M"
    }
    if s in syn:
        return syn[s] if syn[s] in VALID_INTERVALS else None
    return None

# -------------------------
# Utility: get last price
# -------------------------
def get_last_price(symbol: str, interval: str = DEFAULT_INTERVAL) -> Optional[float]:
    df = get_bybit_klines(symbol, interval=interval, limit=2)
    if df.empty:
        return None
    return float(df["close"].iat[-1])

# -------------------------
# Helper: cycle letter
# -------------------------
def get_cycle_letter(cfg: Dict[str, Any], offset: int = 0) -> str:
    forced = cfg.get("forced_letter")
    if forced:
        return forced.upper()
    letters = cfg.get("custom_letters") or "".join(chr(i) for i in range(65, 91))
    letters = list(letters.upper())
    today = datetime.utcnow().date() + timedelta(days=offset)
    idx = (today.timetuple().tm_yday - 1) % len(letters)
    return letters[idx]

# -------------------------
# Telegram handlers
# -------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_chat_config(chat_id)
    await update.message.reply_text("✅ Registered. Daily delivery scheduled. Use /help to see commands.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 tradingBot — commands\n\n"
        "/start - register and schedule daily delivery\n"
        "/help - show this help\n"
        "/status - show config for this chat\n"
        "/list - list today's symbols for the current letter and auto-send charts\n"
        "/chart SYMBOL [INTERVAL] - single candlestick chart (e.g. /chart BTCUSDT 1h)\n"
        "/analyze SYMBOL [INTERVAL] - chart + supply/demand zones\n"
        "/price SYMBOL - live last price\n"
        "/setinterval I - set default kline interval (e.g. 1h,15m,1d or numeric like 60)\n"
        "/sethour H - set UTC hour for daily delivery\n"
        "/setletter X - force today to be letter X\n"
        "/setletters ABC - set custom cycle\n"
        "/setsuffix S or - - set or disable suffix filter (e.g. USDT)\n"
        "/setmin N - set minimum candles required\n"
        "/setmax M - set maximum charts per run\n"
        "/settestpattern a,b,c - set the /test batch pattern (defaults to 20,20,20)\n"
        "/pause - pause daily delivery\n"
        "/resume - resume daily delivery\n"
        "/test [pattern] - RUN auto-listing + auto-charting now\n"
        "/diag SYMBOL - fetch raw Bybit JSON for symbol\n"
        "/export - export chat config as json\n"
        "/import <json> - import chat config\n"
        "/allletters - show full cycle and today's letter\n"
    )
    await update.message.reply_text(help_text)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = ensure_chat_config(update.effective_chat.id)
    await update.message.reply_text(json.dumps(cfg, indent=2))

async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cfg = ensure_chat_config(chat_id)
    letter = get_cycle_letter(cfg)
    await update.message.reply_text(f"📅 Today's letter: {letter}\nFetching Bybit futures symbols...")
    try:
        bybit_all = fetch_bybit_symbols()
        suf = (cfg.get("suffix") or "").upper()
        candidates = [s for s in bybit_all if s.upper().endswith(suf)] if suf else bybit_all
        todays = [s for s in candidates if s.startswith(letter)]
        if not todays:
            await update.message.reply_text(f"No symbols found for letter {letter}")
            return
        joined = "\n".join(todays)
        if len(joined) > 4000:
            await update.message.reply_text(f"Found {len(todays)} symbols — sending first 200")
            await update.message.reply_text("\n".join(todays[:200]))
        else:
            await update.message.reply_text(joined)

        maxc = min(int(cfg.get("max_charts", 6)), MAX_CHARTS_SAFE)
        to_chart = todays[:maxc]
        if to_chart:
            await update.message.reply_text(f"Auto-charting first {len(to_chart)} symbols...")
            loop = __import__("asyncio").get_event_loop()
            per_symbol_delay = 0.8
            for sym in to_chart:
                try:
                    img = await loop.run_in_executor(None, _sync_plot_bytes_safe, sym, cfg.get("interval", DEFAULT_INTERVAL), 50, 1.8)
                    img.seek(0)
                    await update.message.reply_photo(photo=img, caption=f"{sym} {cfg.get('interval', DEFAULT_INTERVAL)}")
                    img.close()
                except Exception as e:
                    logger.exception("Auto-chart failed for %s", sym)
                    await update.message.reply_text(f"Failed to chart {sym}: {e}")
                await __import__("asyncio").sleep(per_symbol_delay)
    except Exception:
        logger.exception("/list error")
        await update.message.reply_text("Error during /list (see logs)")

# Improved cmd_chart using normalize_interval and chat config
async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /chart SYMBOL [INTERVAL]\nExample: /chart ETHUSDT 1h")
        return

    symbol = context.args[0].upper()
    cfg = ensure_chat_config(update.effective_chat.id)

    # determine interval
    if len(context.args) > 1:
        raw_interval = context.args[1]
        interval = normalize_interval(raw_interval)
        if not interval:
            await update.message.reply_text(
                f"Invalid interval '{raw_interval}'. Valid: {', '.join(sorted(VALID_INTERVALS))}.\n"
                "Examples: 15m,1h,1d or numeric 15 => 15m, 60 => 1h."
            )
            return
    else:
        interval = cfg.get('interval', DEFAULT_INTERVAL)

    await update.message.reply_text(f"Fetching {symbol} {interval} from Bybit...")
    try:
        loop = __import__("asyncio").get_event_loop()
        img = await loop.run_in_executor(None, _sync_plot_bytes_safe, symbol, interval, 50, 1.8)
        img.seek(0)
        await update.message.reply_photo(photo=img, caption=f"{symbol} {interval}")
        img.close()
    except Exception as e:
        logger.exception("/chart error: %s", e)
        await update.message.reply_text(f"Failed to chart {symbol}: {e}")

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_chart(update, context)

async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /price SYMBOL")
        return
    symbol = context.args[0].upper()
    price = get_last_price(symbol, interval=DEFAULT_INTERVAL)
    if price is None:
        await update.message.reply_text("Could not fetch price for that symbol.")
    else:
        await update.message.reply_text(f"{symbol} last price: {price}")

# -------------------------
# config setters with normalization fix
# -------------------------
async def cmd_setinterval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setinterval 1h (valid: " + ", ".join(sorted(VALID_INTERVALS)) + ")\nExamples: /setinterval 15m  OR  /setinterval 60")
        return

    raw = context.args[0]
    norm = normalize_interval(raw)
    if not norm:
        await update.message.reply_text(
            f"Invalid interval '{raw}'. Valid options: {', '.join(sorted(VALID_INTERVALS))}.\n"
            "Examples: 1m,5m,15m,1h,4h,1d or numeric: 15 (=>15m), 60 (=>1h), 240 (=>4h)."
        )
        return

    cfg = ensure_chat_config(update.effective_chat.id)
    cfg['interval'] = norm
    save_chat_configs()
    logger.info("Chat %s set interval -> %s", update.effective_chat.id, norm)
    await update.message.reply_text(f"✅ Default interval for this chat set to *{norm}*.", parse_mode="Markdown")

async def cmd_sethour(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /sethour 9 (0-23 UTC)")
        return
    try:
        h = int(context.args[0]) % 24
    except Exception:
        await update.message.reply_text("Invalid hour")
        return
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["hour"] = h
    save_chat_configs()
    await update.message.reply_text(f"Delivery hour set to {h}:00 UTC")

async def cmd_setletter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setletter X")
        return
    val = context.args[0].upper()
    if len(val) != 1 or not val.isalpha():
        await update.message.reply_text("Provide single letter A-Z")
        return
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["forced_letter"] = val
    save_chat_configs()
    await update.message.reply_text(f"Forced letter set to {val}")

async def cmd_setletters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setletters ABC...")
        return
    val = context.args[0].upper()
    if not all(c.isalpha() for c in val):
        await update.message.reply_text("Letters must be alphabetic")
        return
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["custom_letters"] = val
    save_chat_configs()
    await update.message.reply_text(f"Custom letters set to: {val}")

async def cmd_setsuffix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setsuffix S or - to disable")
        return
    val = context.args[0].upper()
    cfg = ensure_chat_config(update.effective_chat.id)
    if val == "-":
        cfg["suffix"] = ""
        await update.message.reply_text("Suffix filter disabled")
    else:
        cfg["suffix"] = val
        await update.message.reply_text(f"Suffix set to {val}")
    save_chat_configs()

async def cmd_setmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmin N")
        return
    try:
        n = int(context.args[0])
    except Exception:
        await update.message.reply_text("Invalid number")
        return
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["min_candles"] = max(1, n)
    save_chat_configs()
    await update.message.reply_text(f"Minimum candles set to {cfg['min_candles']}")

async def cmd_setmax(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmax M")
        return
    try:
        m = int(context.args[0])
    except Exception:
        await update.message.reply_text("Invalid number")
        return
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["max_charts"] = max(0, min(MAX_CHARTS_SAFE, m))
    save_chat_configs()
    await update.message.reply_text(f"Max charts per run set to {cfg['max_charts']}")

async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["paused"] = True
    save_chat_configs()
    await update.message.reply_text("Daily delivery paused for this chat")

async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["paused"] = False
    save_chat_configs()
    await update.message.reply_text("Daily delivery resumed for this chat")

async def cmd_settestpattern(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /settestpattern a,b,c")
        return
    raw = "".join(context.args)
    parts = [p.strip() for p in raw.split(",") if p.strip().isdigit()]
    if not parts:
        await update.message.reply_text("Invalid pattern. Provide comma-separated integers, e.g., 20,20,20")
        return
    pattern = [int(p) for p in parts]
    cfg = ensure_chat_config(update.effective_chat.id)
    cfg["test_pattern"] = pattern
    save_chat_configs()
    await update.message.reply_text(f"Test pattern saved: {pattern}")

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = ensure_chat_config(update.effective_chat.id)
    await update.message.reply_text(json.dumps(cfg))

async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /import <json>")
        return
    payload = " ".join(context.args)
    try:
        obj = json.loads(payload)
        if not isinstance(obj, dict):
            raise ValueError("payload not an object")
        chat_id = str(update.effective_chat.id)
        _chat_configs[chat_id] = obj
        save_chat_configs()
        await update.message.reply_text("Config imported")
    except Exception as e:
        await update.message.reply_text(f"Import failed: {e}")

async def cmd_allletters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = ensure_chat_config(update.effective_chat.id)
    letters = cfg.get("custom_letters") or "".join([chr(i) for i in range(65, 91)])
    today = get_cycle_letter(cfg)
    await update.message.reply_text(f"Letters: {letters}\nToday: {today}")

# -------------------------
# /test command
# -------------------------
async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cfg = ensure_chat_config(chat_id)
    pattern = cfg.get("test_pattern", DEFAULT_TEST_PATTERN)
    if context.args:
        raw = "".join(context.args)
        parts = [p.strip() for p in raw.split(",") if p.strip().isdigit()]
        if parts:
            pattern = [int(p) for p in parts]
            cfg["test_pattern"] = pattern
            save_chat_configs()
    await update.message.reply_text(f"Starting test run with batch pattern {pattern}...")
    try:
        bybit_all = fetch_bybit_symbols()
        suf = (cfg.get("suffix") or "").upper()
        candidates = [s for s in bybit_all if s.upper().endswith(suf)] if suf else bybit_all
        letter = get_cycle_letter(cfg)
        todays = [s for s in candidates if s.startswith(letter)]
        if not todays:
            await update.message.reply_text("No symbols for today")
            return
        idx = 0
        loop = __import__("asyncio").get_event_loop()
        for batch_size in pattern:
            batch = todays[idx: idx + batch_size]
            if not batch:
                break
            await update.message.reply_text(f"Processing batch of {len(batch)} symbols...")
            for s in batch:
                try:
                    img = await loop.run_in_executor(None, _sync_plot_bytes_safe, s, cfg.get("interval", DEFAULT_INTERVAL), 50, 1.8)
                    img.seek(0)
                    await update.message.reply_photo(photo=img, caption=f"{s} {cfg.get('interval', DEFAULT_INTERVAL)}")
                    img.close()
                except Exception as e:
                    logger.exception("Test chart failed for %s", s)
                    await update.message.reply_text(f"Failed to chart {s}: {e}")
            idx += batch_size
            await __import__("asyncio").sleep(1)
        await update.message.reply_text("Test run complete")
    except Exception:
        logger.exception("Test command failed")
        await update.message.reply_text("Test failed (see logs)")

# -------------------------
# /diag command
# -------------------------
async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /diag SYMBOL")
        return
    symbol = context.args[0].upper()
    try:
        url = BYBIT_BASE + "/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": MAP_INTERVAL.get(DEFAULT_INTERVAL, "60"), "limit": "10"}
        data = http_get_json(url, params=params)
        s = json.dumps(data, indent=2)
        if len(s) > 3800:
            s = s[:3800] + "\n\n...[truncated]"
        await update.message.reply_text(f"Raw Bybit kline JSON for {symbol}:\n{s}")
    except Exception as e:
        logger.exception("Diag failed for %s", symbol)
        await update.message.reply_text(f"Diag failed: {e}")

# -------------------------
# Register handlers
# -------------------------
def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("chart", cmd_chart))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("setinterval", cmd_setinterval))
    app.add_handler(CommandHandler("sethour", cmd_sethour))
    app.add_handler(CommandHandler("setletter", cmd_setletter))
    app.add_handler(CommandHandler("setletters", cmd_setletters))
    app.add_handler(CommandHandler("setsuffix", cmd_setsuffix))
    app.add_handler(CommandHandler("setmin", cmd_setmin))
    app.add_handler(CommandHandler("setmax", cmd_setmax))
    app.add_handler(CommandHandler("settestpattern", cmd_settestpattern))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("test", cmd_test))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("allletters", cmd_allletters))
    app.add_handler(CommandHandler("diag", cmd_diag))

# -------------------------
# Scheduler thread
# -------------------------
def scheduler_entry(app_getter):
    logger.info("Scheduler thread started")
    while True:
        try:
            app = app_getter()
            if app is None or not hasattr(app, "bot"):
                time.sleep(1)
                continue
            now = datetime.utcnow()
            for chat_id_str, cfg in list(_chat_configs.items()):
                try:
                    if cfg.get("paused"):
                        continue
                    hour = int(cfg.get("hour", DEFAULT_HOUR_UTC))
                    last_run = cfg.get("last_run")
                    if last_run == now.date().isoformat():
                        continue
                    if now.hour == hour:
                        bybit_all = fetch_bybit_symbols()
                        suf = (cfg.get("suffix") or "").upper()
                        candidates = [s for s in bybit_all if s.upper().endswith(suf)] if suf else bybit_all
                        letter = get_cycle_letter(cfg)
                        todays = [s for s in candidates if s.startswith(letter)]
                        if not todays:
                            app.bot.send_message(chat_id=int(chat_id_str), text=f"No symbols for letter {letter} today")
                        else:
                            text = f"Daily letter {letter}:\n" + "\n".join(todays[:200])
                            app.bot.send_message(chat_id=int(chat_id_str), text=text)
                            maxc = min(int(cfg.get("max_charts", 6)), MAX_CHARTS_SAFE)
                            for s in todays[:maxc]:
                                try:
                                    df = get_bybit_klines(s, interval=cfg.get("interval", DEFAULT_INTERVAL), limit=DEFAULT_LIMIT)
                                    if df.empty:
                                        app.bot.send_message(chat_id=int(chat_id_str), text=f"No data for {s}")
                                        continue
                                    img = plotly_candles_with_zones_light(df, s, cfg.get("interval", DEFAULT_INTERVAL))
                                    img.seek(0)
                                    app.bot.send_photo(chat_id=int(chat_id_str), photo=img, caption=f"{s} {cfg.get('interval', DEFAULT_INTERVAL)}")
                                    img.close()
                                except Exception:
                                    logger.exception("Daily chart failed for %s", s)
                                    app.bot.send_message(chat_id=int(chat_id_str), text=f"Failed to chart {s}")
                        cfg["last_run"] = now.date().isoformat()
                        save_chat_configs()
                except Exception:
                    logger.exception("Scheduler per-chat error")
            time.sleep(max(1, 60 - datetime.utcnow().second))
        except Exception:
            logger.exception("Scheduler top-level error")
            time.sleep(5)

# -------------------------
# Build application & supervisor loop
# -------------------------
_app: Optional[Application] = None

def build_application() -> Application:
    global _app
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    register_handlers(app)
    _app = app
    return app

def get_app_ref() -> Optional[Application]:
    return _app

def main_forever():
    # start scheduler thread
    sched_thread = threading.Thread(target=scheduler_entry, args=(get_app_ref,), daemon=True)
    sched_thread.start()

    logger.info("Starting supervisor loop (auto-restart).")
    while True:
        try:
            app = build_application()
            logger.info("Launching Application.run_polling()")
            app.run_polling()
            logger.info("Application.run_polling() finished normally; exiting.")
            break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - exiting supervisor.")
            break
        except Exception:
            logger.exception("Bot crashed; restarting in 5 seconds...")
            try:
                global _app
                _app = None
            except Exception:
                pass
            time.sleep(5)
            continue

if __name__ == "__main__":
    try:
        main_forever()
    except Exception:
        logger.exception("Fatal error in main entrypoint")
        sys.exit(1)
