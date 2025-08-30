#!/usr/bin/env python3
"""
tradingbot.py
Single-file Telegram trading-chart bot with many commands:
- /start, /help, /status, /list, /chart, /analyze, /price, /setinterval, /sethour, /setletter,
  /setletters, /setsuffix, /setmin, /setmax, /settestpattern, /pause, /resume, /test, /diag,
  /export, /import, /allletters
Uses Bybit public API when available, falls back to Binance public API.
Generates Plotly candlestick charts and returns PNGs using kaleido.
Persists per-chat config to chat_config.json.
"""

import os
import json
import math
import asyncio
import logging
import traceback
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import List, Dict, Optional, Any

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# CONFIG / ENV
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
CHAT_CONFIG_PATH = "chat_config.json"
DEFAULT_INTERVAL = "1h"
DEFAULT_HOUR_UTC = 12
DEFAULT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_SUFFIX = "USDT"
DEFAULT_MIN_CANDLES = 50
DEFAULT_MAX_CHARTS = 20
DEFAULT_TEST_PATTERN = [20, 20, 20]

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingBot_superlong")

# ensure token present
if not TELEGRAM_TOKEN:
    log.error("TELEGRAM_TOKEN env var is required. Exiting.")
    raise SystemExit("TELEGRAM_TOKEN env var is required.")

# simple file-backed chat config store
def load_configs() -> Dict[str, Any]:
    if os.path.exists(CHAT_CONFIG_PATH):
        try:
            with open(CHAT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.exception("Failed to load chat_config.json, starting fresh.")
            return {}
    return {}

def save_configs(cfgs: Dict[str, Any]) -> None:
    with open(CHAT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfgs, f, indent=2)

configs = load_configs()

def get_chat_cfg(chat_id: str) -> Dict[str, Any]:
    if chat_id not in configs:
        configs[chat_id] = {
            "interval": DEFAULT_INTERVAL,
            "hour_utc": DEFAULT_HOUR_UTC,
            "letters": DEFAULT_LETTERS,
            "suffix": DEFAULT_SUFFIX,
            "min_candles": DEFAULT_MIN_CANDLES,
            "max_charts": DEFAULT_MAX_CHARTS,
            "test_pattern": DEFAULT_TEST_PATTERN,
            "paused": False,
        }
        save_configs(configs)
    return configs[chat_id]

# networking helpers with retries
def http_get_json(url: str, params: dict = None, headers: dict = None, max_retries=4, timeout=10):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                last_exc = Exception(f"HTTP {r.status_code}: {r.text[:400]}")
                log.warning("%s returned %s (attempt %d/%d)", url, r.status_code, attempt, max_retries)
                # If 403 country block or similar, surface quickly
                if r.status_code in (401, 403):
                    break
        except Exception as e:
            last_exc = e
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt, max_retries, e)
    raise RuntimeError(f"Failed GET {url} after {max_retries} attempts: {last_exc}")

# Bybit public endpoints (v5)
def fetch_bybit_instruments():
    url = "https://api.bybit.com/v5/market/instruments-info"
    try:
        data = http_get_json(url)
        # format: data['result']['list']...
        lst = []
        for item in data.get("result", {}).get("list", []):
            symbol = item.get("symbol")
            if symbol:
                lst.append(symbol)
        return lst
    except Exception as e:
        log.warning("Bybit instruments fetch failed: %s", e)
        return []

def bybit_klines(symbol: str, interval: str, limit: int = 500):
    """
    Try Bybit v5 kline. If fails or blocked, raises exception.
    interval examples: '1', '1h', '4h', '1d' -> Bybit expects '1', '60', '240', 'D' sometimes.
    We'll map common: '1m','3m','5m','15m','30m','1h','4h','1d' to Bybit interval codes if needed.
    But Bybit supports the same strings as Binance in v5: '1m','3m','5m','15m','30m','1h','4h','1d'
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get_json(url, params=params)
    # bybit returns data['result']['list'] as rows: [open_time, open, high, low, close, volume, ...]
    res = []
    for row in data.get("result", {}).get("list", []):
        # ensure correct shape
        t = int(row[0]) // 1000 if row[0] > 1e12 else int(row[0])
        res.append([datetime.fromtimestamp(t, tz=timezone.utc), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    df = pd.DataFrame(res, columns=["time", "open", "high", "low", "close", "volume"])
    df.set_index("time", inplace=True)
    return df

# Binance fallback
def binance_klines(symbol: str, interval: str, limit: int = 500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    rows = r.json()
    res = []
    for row in rows:
        t = int(row[0]) // 1000
        res.append([datetime.fromtimestamp(t, tz=timezone.utc), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    df = pd.DataFrame(res, columns=["time", "open", "high", "low", "close", "volume"])
    df.set_index("time", inplace=True)
    return df

def fetch_klines_with_fallback(symbol: str, interval: str, limit: int = 500):
    """
    Try Bybit first. If blocked/fails, fallback to Binance.
    """
    # If symbol like 'BTCUSDT' or 'ETHUSDT' it's fine for both.
    # Try Bybit
    try:
        df = bybit_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning("Bybit kline failed for %s: %s", symbol, e)

    # Try Binance
    try:
        df = binance_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning("Binance kline failed for %s: %s", symbol, e)

    # Last fallback: try yfinance for some symbols (best-effort)
    try:
        import yfinance as yf
        yf_symbol = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        hist = yf.download(yf_symbol, period="60d", interval=interval if interval.endswith("m") or interval.endswith("h") or interval.endswith("d") else "1d", progress=False)
        if not hist.empty:
            hist = hist.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            hist.index = [pd.Timestamp(t).tz_convert('UTC') if hasattr(t, "tzinfo") else pd.Timestamp(t).tz_localize('UTC') for t in hist.index]
            return hist[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        log.debug("yfinance fallback failed: %s", e)

    raise RuntimeError(f"No kline source returned data for {symbol} {interval}")

# Price fetch with fallback
def fetch_price(symbol: str) -> float:
    # try Bybit ticker
    try:
        url = "https://api.bybit.com/v2/public/tickers"
        r = requests.get(url, params={"symbol": symbol}, timeout=6)
        if r.status_code == 200:
            j = r.json()
            ret = j.get("result", [])
            if ret:
                price = float(ret[0].get("last_price"))
                return price
    except Exception:
        pass
    # try Binance
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        r = requests.get(url, params={"symbol": symbol}, timeout=6)
        r.raise_for_status()
        j = r.json()
        return float(j["price"])
    except Exception:
        pass
    # yfinance fallback
    try:
        import yfinance as yf
        yf_symbol = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        t = yf.Ticker(yf_symbol)
        info = t.history(period="1d")
        if not info.empty:
            return float(info["Close"].iloc[-1])
    except Exception:
        pass
    raise RuntimeError(f"Unable to fetch price for {symbol}")

# Plotting utilities
def _sync_plot_bytes(df: pd.DataFrame, symbol: str, width=1200, height=720, show_volume=True):
    """
    Synchronous: return PNG bytes from a Plotly candlestick chart built from df (index datetime).
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol
    ))
    if show_volume and "volume" in df.columns:
        # Add a secondary y-axis for volume
        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=False)),
            margin=dict(l=10, r=10, t=40, b=20),
            height=height,
            width=width
        )
        # Add volume as bar trace
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], yaxis="y2", name="Volume", opacity=0.3))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1.02, title="Volume")
        )
    else:
        fig.update_layout(height=height, width=width, margin=dict(l=10, r=10, t=40, b=20))
    fig.update_layout(title=f"{symbol} • Candlestick", template="plotly_white")
    # export to png bytes using kaleido
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return img_bytes

# Simple supply & demand zone identification (very basic)
def compute_supply_demand_zones(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
    """
    Finds local tops (supply) and bottoms (demand) as candidate zones.
    Very naive: pick N highest highs and N lowest lows over lookback window.
    """
    highs = df["high"].tail(lookback)
    lows = df["low"].tail(lookback)
    supply = sorted(highs.nlargest(3).tolist())
    demand = sorted(lows.nsmallest(3).tolist())
    return {"supply": supply, "demand": demand}

# Chat scheduling with APScheduler
scheduler = BackgroundScheduler()
scheduler.start()

def schedule_daily_for_chat(chat_id: str):
    cfg = get_chat_cfg(str(chat_id))
    hour = cfg.get("hour_utc", DEFAULT_HOUR_UTC)
    # remove any existing jobs for that chat
    job_id = f"daily_{chat_id}"
    scheduler.remove_job(job_id=job_id, jobstore=None) if scheduler.get_job(job_id) else None
    # schedule at specified UTC hour, daily
    trigger = CronTrigger(hour=hour, minute=0, timezone=timezone.utc)
    scheduler.add_job(func=partial(run_daily_delivery, chat_id), trigger=trigger, id=job_id, replace_existing=True)
    log.info("Scheduled daily delivery for chat %s at UTC hour %d", chat_id, hour)

async def run_daily_delivery(chat_id: str):
    """
    Build today's list and send charts (respecting paused flag and config).
    Runs inside scheduler (sync) — but we will call into async via application.
    """
    # This function will be executed by APScheduler in a thread; use Application to send messages.
    try:
        cfg = get_chat_cfg(str(chat_id))
        if cfg.get("paused", False):
            log.info("Delivery for chat %s is paused.", chat_id)
            return
        # determine today's letter from cycle
        letters = cfg.get("letters", DEFAULT_LETTERS)
        idx = (datetime.utcnow().date().toordinal()) % len(letters)
        today_letter = letters[idx]
        suffix = cfg.get("suffix", DEFAULT_SUFFIX)
        max_charts = cfg.get("max_charts", DEFAULT_MAX_CHARTS)
        # fetch symbols — try Bybit first
        symbols = fetch_bybit_instruments()
        if not symbols:
            log.info("Bybit instruments empty; trying Binance exchangeInfo")
            try:
                j = http_get_json("https://api.binance.com/api/v3/exchangeInfo")
                symbols = [s["symbol"] for s in j.get("symbols", [])]
            except Exception:
                log.warning("Failed to fetch exchange symbols from Binance.")
                symbols = []
        # filter by letter and suffix
        filtered = [s for s in symbols if s.upper().startswith(today_letter.upper()) and s.upper().endswith(suffix.upper())]
        filtered = filtered[:max_charts]
        a = application  # global from main
        # send message header
        await a.bot.send_message(int(chat_id), text=f"Daily charts for letter {today_letter} ({len(filtered)} symbols).")
        # for each symbol, build a chart and send
        for sym in filtered:
            try:
                df = fetch_klines_with_fallback(sym, cfg.get("interval", DEFAULT_INTERVAL), limit=200)
                if df.shape[0] < cfg.get("min_candles", DEFAULT_MIN_CANDLES):
                    await a.bot.send_message(int(chat_id), text=f"Skipping {sym}: insufficient candles ({df.shape[0]})")
                    continue
                # render plot in executor to avoid blocking scheduler thread
                loop = asyncio.get_running_loop()
                img_bytes = await loop.run_in_executor(None, partial(_sync_plot_bytes, df.tail(200), sym))
                await a.bot.send_photo(int(chat_id), photo=img_bytes, caption=f"{sym} • {cfg.get('interval')}")
            except Exception as e:
                await a.bot.send_message(int(chat_id), text=f"Failed chart {sym}: {e}")
    except Exception:
        log.exception("run_daily_delivery failed")

# Telegram command handlers
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    get_chat_cfg(chat_id)
    save_configs(configs)
    schedule_daily_for_chat(chat_id)
    await update.message.reply_text("Registered for daily delivery. Use /help to see commands.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = """TradingBot commands:
/start - register and schedule daily delivery
/help - show this help
/status - show config for this chat
/list - list today's symbols for the current letter
/chart SYMBOL [INTERVAL] - single candlestick chart (e.g. /chart BTCUSDT 1h)
/analyze SYMBOL [INTERVAL] - chart + simple supply/demand zones
/price SYMBOL - live last price
/setinterval I - set default kline interval (e.g. 1h,15m,1d or numeric like 60)
/sethour H - set UTC hour for daily delivery
/setletter X - force today to be letter X (single letter)
//setletters ABC - set custom cycle
/setsuffix S or - - set or disable suffix filter (e.g. USDT or - to disable)
/setmin N - set minimum candles required
/setmax M - set maximum charts per run
/settestpattern a,b,c - set the /test batch pattern (defaults to 20,20,20)
/pause - pause daily delivery
/resume - resume daily delivery
/test [pattern] - RUN auto-listing + auto-charting now
/diag SYMBOL - fetch raw Bybit JSON for symbol
/export - export chat config as json
/import <json> - import chat config (paste JSON)
/allletters - show full cycle and today's letter
"""
    await update.message.reply_text(txt)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    idx = (datetime.utcnow().date().toordinal()) % len(cfg.get("letters", DEFAULT_LETTERS))
    today_letter = cfg.get("letters", DEFAULT_LETTERS)[idx]
    text = (
        f"Config for chat {chat_id}:\n"
        f"interval: {cfg.get('interval')}\n"
        f"hour_utc: {cfg.get('hour_utc')}\n"
        f"letters: {cfg.get('letters')}\n"
        f"today's letter: {today_letter}\n"
        f"suffix: {cfg.get('suffix')}\n"
        f"min_candles: {cfg.get('min_candles')}\n"
        f"max_charts: {cfg.get('max_charts')}\n"
        f"test_pattern: {cfg.get('test_pattern')}\n"
        f"paused: {cfg.get('paused')}\n"
    )
    await update.message.reply_text(text)

async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    letters = cfg.get("letters", DEFAULT_LETTERS)
    idx = (datetime.utcnow().date().toordinal()) % len(letters)
    today_letter = letters[idx]
    suffix = cfg.get("suffix", DEFAULT_SUFFIX)
    # fetch symbols
    symbols = fetch_bybit_instruments()
    if not symbols:
        try:
            j = http_get_json("https://api.binance.com/api/v3/exchangeInfo")
            symbols = [s["symbol"] for s in j.get("symbols", [])]
        except Exception:
            symbols = []
    filtered = [s for s in symbols if s.upper().startswith(today_letter.upper()) and (not suffix or s.upper().endswith(suffix.upper()))]
    await update.message.reply_text(f"Today's letter {today_letter}. Found {len(filtered)} symbols (showing up to {cfg.get('max_charts')}).\n" + ", ".join(filtered[:cfg.get("max_charts")]))

async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /chart SYMBOL [INTERVAL]")
        return
    symbol = context.args[0].upper()
    interval = context.args[1] if len(context.args) > 1 else get_chat_cfg(str(update.effective_chat.id)).get("interval", DEFAULT_INTERVAL)
    try:
        df = fetch_klines_with_fallback(symbol, interval, limit=500)
        if df.empty:
            raise RuntimeError("No kline data")
        # ensure min candles
        if df.shape[0] < get_chat_cfg(str(update.effective_chat.id)).get("min_candles", DEFAULT_MIN_CANDLES):
            await update.message.reply_text(f"Insufficient candles ({df.shape[0]}) for {symbol}")
            return
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, partial(_sync_plot_bytes, df.tail(200), symbol))
        await update.message.reply_photo(photo=img, caption=f"{symbol} • {interval}")
    except Exception as e:
        await update.message.reply_text(f"Error charting {symbol}: {e}")

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /analyze SYMBOL [INTERVAL]")
        return
    symbol = context.args[0].upper()
    interval = context.args[1] if len(context.args) > 1 else get_chat_cfg(str(update.effective_chat.id)).get("interval", DEFAULT_INTERVAL)
    try:
        df = fetch_klines_with_fallback(symbol, interval, limit=500)
        if df.shape[0] < 20:
            await update.message.reply_text(f"Insufficient data for {symbol}")
            return
        zones = compute_supply_demand_zones(df, lookback=200)
        # draw chart + horizontal lines for zones
        def _plot_with_zones():
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol))
            for s in zones.get("supply", []):
                fig.add_hline(y=s, line=dict(dash="dash"), annotation_text=f"supply {s:.4f}")
            for d in zones.get("demand", []):
                fig.add_hline(y=d, line=dict(dash="dot"), annotation_text=f"demand {d:.4f}")
            fig.update_layout(title=f"{symbol} • Analyze ({interval})", height=720, width=1200)
            return fig.to_image(format="png", engine="kaleido")
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, _plot_with_zones)
        await update.message.reply_photo(photo=img, caption=f"{symbol} • zones (supply/demand)")
    except Exception as e:
        await update.message.reply_text(f"Error analyzing {symbol}: {e}")

async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /price SYMBOL")
        return
    symbol = context.args[0].upper()
    try:
        price = fetch_price(symbol)
        await update.message.reply_text(f"{symbol} price: {price}")
    except Exception as e:
        await update.message.reply_text(f"Failed to fetch price for {symbol}: {e}")

# setters
async def cmd_setinterval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setinterval INTERVAL (e.g. 1h,15m,1d)")
        return
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    cfg["interval"] = context.args[0]
    save_configs(configs)
    await update.message.reply_text(f"Interval set to {cfg['interval']}")

async def cmd_sethour(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /sethour H (UTC hour 0-23)")
        return
    try:
        h = int(context.args[0]) % 24
    except Exception:
        await update.message.reply_text("Invalid hour.")
        return
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    cfg["hour_utc"] = h
    save_configs(configs)
    schedule_daily_for_chat(chat_id)
    await update.message.reply_text(f"UTC hour set to {h}")

async def cmd_setletter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setletter X")
        return
    c = context.args[0].upper()[0]
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    # make today's letter be this for immediate operations by shoving a one-letter letters string
    cfg["letters"] = c
    save_configs(configs)
    schedule_daily_for_chat(chat_id)
    await update.message.reply_text(f"Today's letter forced to {c}")

async def cmd_setletters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setletters ABC... (cycle)")
        return
    s = "".join(context.args[0].upper())
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    cfg["letters"] = s
    save_configs(configs)
    schedule_daily_for_chat(chat_id)
    await update.message.reply_text(f"Letters cycle set to {s}")

async def cmd_setsuffix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setsuffix S (e.g. USDT) or - to disable")
        return
    s = context.args[0].upper()
    if s == "-":
        s = ""
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    cfg["suffix"] = s
    save_configs(configs)
    await update.message.reply_text(f"Suffix set to '{s}'")

async def cmd_setmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmin N")
        return
    try:
        n = int(context.args[0])
    except Exception:
        await update.message.reply_text("Invalid integer")
        return
    cfg = get_chat_cfg(str(update.effective_chat.id))
    cfg["min_candles"] = n
    save_configs(configs)
    await update.message.reply_text(f"min_candles set to {n}")

async def cmd_setmax(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmax M")
        return
    try:
        m = int(context.args[0])
    except Exception:
        await update.message.reply_text("Invalid integer")
        return
    cfg = get_chat_cfg(str(update.effective_chat.id))
    cfg["max_charts"] = m
    save_configs(configs)
    await update.message.reply_text(f"max_charts set to {m}")

async def cmd_settestpattern(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /settestpattern a,b,c")
        return
    try:
        pattern = [int(x) for x in context.args[0].split(",")]
    except Exception:
        await update.message.reply_text("Invalid pattern. Use comma separated integers.")
        return
    cfg = get_chat_cfg(str(update.effective_chat.id))
    cfg["test_pattern"] = pattern
    save_configs(configs)
    await update.message.reply_text(f"test_pattern set to {pattern}")

async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = get_chat_cfg(str(update.effective_chat.id))
    cfg["paused"] = True
    save_configs(configs)
    await update.message.reply_text("Daily delivery paused.")

async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = get_chat_cfg(str(update.effective_chat.id))
    cfg["paused"] = False
    save_configs(configs)
    await update.message.reply_text("Daily delivery resumed.")

# test runner: runs listing + charting in batches (non-blocking)
async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    pattern = cfg.get("test_pattern", DEFAULT_TEST_PATTERN)
    if context.args:
        try:
            pattern = [int(x) for x in context.args[0].split(",")]
        except Exception:
            await update.message.reply_text("Invalid pattern argument; using saved pattern.")
    # fetch symbols
    symbols = fetch_bybit_instruments()
    if not symbols:
        try:
            j = http_get_json("https://api.binance.com/api/v3/exchangeInfo")
            symbols = [s["symbol"] for s in j.get("symbols", [])]
        except Exception:
            symbols = []
    # filter by today's letter
    letters = cfg.get("letters", DEFAULT_LETTERS)
    idx = (datetime.utcnow().date().toordinal()) % len(letters)
    today_letter = letters[idx]
    suffix = cfg.get("suffix", DEFAULT_SUFFIX)
    filtered = [s for s in symbols if s.upper().startswith(today_letter.upper()) and (not suffix or s.upper().endswith(suffix.upper()))]
    await update.message.reply_text(f"Starting test run with batch pattern {pattern}...\nFound {len(filtered)} symbols for {today_letter}.")
    # iterate batches
    pos = 0
    a = application
    for batch_size in pattern:
        if pos >= len(filtered):
            break
        batch = filtered[pos:pos+batch_size]
        pos += batch_size
        await update.message.reply_text(f"Processing batch of {len(batch)} symbols...")
        for sym in batch:
            try:
                df = fetch_klines_with_fallback(sym, cfg.get("interval", DEFAULT_INTERVAL), limit=200)
                if df.shape[0] < cfg.get("min_candles", DEFAULT_MIN_CANDLES):
                    await update.message.reply_text(f"Failed to chart {sym}: No kline data or insufficient candles ({df.shape[0]})")
                    continue
                loop = asyncio.get_running_loop()
                img = await loop.run_in_executor(None, partial(_sync_plot_bytes, df.tail(200), sym))
                await update.message.reply_photo(photo=img, caption=f"{sym} • test")
            except Exception as e:
                await update.message.reply_text(f"Failed to chart {sym}: {e}")
    await update.message.reply_text("Test run completed.")

# diag - return raw Bybit JSON if possible
async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /diag SYMBOL")
        return
    symbol = context.args[0].upper()
    try:
        url = "https://api.bybit.com/v5/market/instruments-info"
        j = http_get_json(url, params={"symbol": symbol})
        await update.message.reply_text(json.dumps(j, indent=2)[:4000])
    except Exception as e:
        await update.message.reply_text(f"Diag fetch failed: {e}")

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    await update.message.reply_text("Chat config JSON:\n" + json.dumps(cfg, indent=2))

async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Expect user to paste JSON after command
    chat_id = str(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("Usage: /import <json>")
        return
    try:
        incoming = " ".join(context.args)
        new_cfg = json.loads(incoming)
        configs[chat_id] = new_cfg
        save_configs(configs)
        schedule_daily_for_chat(chat_id)
        await update.message.reply_text("Imported config.")
    except Exception as e:
        await update.message.reply_text(f"Import failed: {e}")

async def cmd_allletters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = get_chat_cfg(chat_id)
    letters = cfg.get("letters", DEFAULT_LETTERS)
    idx = (datetime.utcnow().date().toordinal()) % len(letters)
    today_letter = letters[idx]
    await update.message.reply_text(f"Letters cycle: {letters}\nToday's letter: {today_letter}")

# catch-all error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Exception while handling an update: %s", context.error)
    try:
        tb = "".join(traceback.format_exception(None, context.error, context.error.__traceback__))
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {context.error}\n{tb[:1000]}")
    except Exception:
        log.exception("Failed to send error message to chat")

# wire up Application
application = Application.builder().token(TELEGRAM_TOKEN).build()

# Register handlers
application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CommandHandler("help", cmd_help))
application.add_handler(CommandHandler("status", cmd_status))
application.add_handler(CommandHandler("list", cmd_list))
application.add_handler(CommandHandler("chart", cmd_chart))
application.add_handler(CommandHandler("analyze", cmd_analyze))
application.add_handler(CommandHandler("price", cmd_price))
application.add_handler(CommandHandler("setinterval", cmd_setinterval))
application.add_handler(CommandHandler("sethour", cmd_sethour))
application.add_handler(CommandHandler("setletter", cmd_setletter))
application.add_handler(CommandHandler("setletters", cmd_setletters))
application.add_handler(CommandHandler("setsuffix", cmd_setsuffix))
application.add_handler(CommandHandler("setmin", cmd_setmin))
application.add_handler(CommandHandler("setmax", cmd_setmax))
application.add_handler(CommandHandler("settestpattern", cmd_settestpattern))
application.add_handler(CommandHandler("pause", cmd_pause))
application.add_handler(CommandHandler("resume", cmd_resume))
application.add_handler(CommandHandler("test", cmd_test))
application.add_handler(CommandHandler("diag", cmd_diag))
application.add_handler(CommandHandler("export", cmd_export))
application.add_handler(CommandHandler("import", cmd_import))
application.add_handler(CommandHandler("allletters", cmd_allletters))
application.add_error_handler(error_handler)

# On startup: schedule existing chat deliveries
def startup_schedule_all():
    for chat_id in list(configs.keys()):
        try:
            schedule_daily_for_chat(chat_id)
        except Exception:
            log.exception("Failed to schedule for chat %s", chat_id)

if __name__ == "__main__":
    log.info("Scheduler thread started")
    startup_schedule_all()
    # Start polling (telemetry printed in logs)
    log.info("Launching Application.run_polling()")
    application.run_polling()
    log.info("Exited")

