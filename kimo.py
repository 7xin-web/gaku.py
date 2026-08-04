# ==========================================
# FILE: fyousou.py (修復・完全版)
# ==========================================

"""
GMO / SBI FX クオンツAI予測ダッシュボード (fyousou.py - 完全独立スタンドアロン版)
Yahoo Financeから主要10通貨ペアデータをリアルタイム取得し、
機械学習(RandomForest)とテクニカル指標(SMA, RSI, MACD, ATR, Bollinger)から目標pips到達確率を算出・可視化します。
【新機能】予測精度トラッカー(Prediction Tracker) & 継続的再学習(Continual Learning)を搭載。
"""

import os
import json
import logging
import math
import smtplib
import urllib.request
import urllib.parse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
import plotly.graph_objects as go

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. SBI / GMO為替メインレート 10通貨ペアの定義
SBI_PAIRS = [
    {"symbol": "AUDUSD=X", "ticker": "AUDUSD=X", "name": "AUD/USD", "disp": "豪ドル/米ドル", "type": "USD", "pip_scale": 0.0001, "target_pips": 250},
    {"symbol": "USDJPY=X", "ticker": "USDJPY=X", "name": "USD/JPY", "disp": "米ドル/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "EURJPY=X", "ticker": "EURJPY=X", "name": "EUR/JPY", "disp": "ユーロ/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "GBPJPY=X", "ticker": "GBPJPY=X", "name": "GBP/JPY", "disp": "ポンド/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "AUDJPY=X", "ticker": "AUDJPY=X", "name": "AUD/JPY", "disp": "豪ドル/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "NZDJPY=X", "ticker": "NZDJPY=X", "name": "NZD/JPY", "disp": "NZドル/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "CADJPY=X", "ticker": "CADJPY=X", "name": "CAD/JPY", "disp": "カナダドル/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "CHFJPY=X", "ticker": "CHFJPY=X", "name": "CHF/JPY", "disp": "スイスフラン/円", "type": "JPY", "pip_scale": 0.01, "target_pips": 250},
    {"symbol": "GBPUSD=X", "ticker": "GBPUSD=X", "name": "GBP/USD", "disp": "ポンド/米ドル", "type": "USD", "pip_scale": 0.0001, "target_pips": 250},
    {"symbol": "EURUSD=X", "ticker": "EURUSD=X", "name": "EUR/USD", "disp": "ユーロ/米ドル", "type": "USD", "pip_scale": 0.0001, "target_pips": 250}
]

GMO_PAIRS = {pair["name"]: pair for pair in SBI_PAIRS}
TARGET_PAIRS = SBI_PAIRS
HISTORY_FILE = "prediction_history.csv"


# --- 2. 予測ログ蓄積・答え合わせ (Outcome Evaluator) & CSV管理モジュール ---
def load_prediction_history() -> pd.DataFrame:
    """保存された予測ログ履歴を読み込む"""
    cols = ["id", "timestamp", "pair_name", "price", "target_pips", "pred_direction", "long_prob", "short_prob", "outcome", "evaluated_at"]
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            for col in cols:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            logging.error(f"Failed to load prediction history CSV: {e}")
    return pd.DataFrame(columns=cols)


def save_prediction_history(df: pd.DataFrame):
    """予測ログ履歴をCSVに保存"""
    try:
        df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        logging.error(f"Failed to save prediction history CSV: {e}")


def record_current_predictions(results: list, target_pips: float):
    """現在の予測結果をログに保存"""
    df_hist = load_prediction_history()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []

    for r in results:
        pair_name = r["name"]
        price = r["price"]
        long_p = r["long_prob"]
        short_p = r["short_prob"]

        if long_p >= short_p:
            pred_dir = "Long"
        else:
            pred_dir = "Short"

        if not df_hist.empty:
            recent_same = df_hist[(df_hist["pair_name"] == pair_name) & (df_hist["timestamp"] > (pd.to_datetime(now_str) - pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"))]
            if not recent_same.empty:
                continue

        rec_id = f"{pair_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(df_hist) + len(new_rows) + 1}"
        new_rows.append({
            "id": rec_id,
            "timestamp": now_str,
            "pair_name": pair_name,
            "price": price,
            "target_pips": target_pips,
            "pred_direction": pred_dir,
            "long_prob": long_p,
            "short_prob": short_p,
            "outcome": -1,
            "evaluated_at": ""
        })

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_updated = pd.concat([df_hist, df_new], ignore_index=True)
        save_prediction_history(df_updated)
        return len(new_rows)
    return 0


def evaluate_prediction_outcomes(pairs_data_map: dict) -> pd.DataFrame:
    """過去の未確定予測ログに対し正解判定を行う"""
    df_hist = load_prediction_history()
    if df_hist.empty:
        return df_hist

    updated = False
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in df_hist.iterrows():
        if row["outcome"] != -1 and not pd.isna(row["outcome"]):
            continue

        pair_name = row["pair_name"]
        pred_time = pd.to_datetime(row["timestamp"])
        entry_price = float(row["price"])
        target_pips = float(row["target_pips"]) if not pd.isna(row["target_pips"]) else 250.0
        pred_dir = str(row["pred_direction"])

        pair_config = GMO_PAIRS.get(pair_name)
        if not pair_config:
            continue

        pip_scale = pair_config["pip_scale"]
        target_val = target_pips * pip_scale

        if pair_name in pairs_data_map and not pairs_data_map[pair_name].empty:
            chart_df = pairs_data_map[pair_name]
            sub_df = chart_df[chart_df.index >= pred_time]
            if len(sub_df) < 1:
                continue

            max_high = sub_df["High"].max()
            min_low = sub_df["Low"].min()

            if pred_dir == "Long":
                if (max_high - entry_price) >= target_val:
                    df_hist.at[idx, "outcome"] = 1
                    df_hist.at[idx, "evaluated_at"] = now_str
                    updated = True
                elif len(sub_df) >= 15:
                    df_hist.at[idx, "outcome"] = 0
                    df_hist.at[idx, "evaluated_at"] = now_str
                    updated = True
            elif pred_dir == "Short":
                if (entry_price - min_low) >= target_val:
                    df_hist.at[idx, "outcome"] = 1
                    df_hist.at[idx, "evaluated_at"] = now_str
                    updated = True
                elif len(sub_df) >= 15:
                    df_hist.at[idx, "outcome"] = 0
                    df_hist.at[idx, "evaluated_at"] = now_str
                    updated = True

    if updated:
        save_prediction_history(df_hist)

    return df_hist


# --- 3. メール送信機能 ---
def send_smtp_email(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    receiver_email: str = "huashenfo@gmail.com",
    subject: str = "",
    body_html: str = ""
) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        html_part = MIMEText(body_html, "html", "utf-8")
        msg.attach(html_part)

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())

        logging.info(f"Email successfully sent to {receiver_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False


def build_signal_email_html(signals_df: pd.DataFrame, threshold_pct: float = 65.0) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_html = ""
    for idx, row in signals_df.iterrows():
        is_highlight = row.get("long_prob", 0) >= threshold_pct or row.get("short_prob", 0) >= threshold_pct
        bg_style = "background-color: #f0fdf4;" if is_highlight else ""
        
        rows_html += f"""
        <tr style="{bg_style} border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 10px; font-weight: bold;">{row.get('name', '')} ({row.get('display_name', '')})</td>
            <td style="padding: 10px; color: #2563eb; font-weight: bold;">{row.get('price', 0)}</td>
            <td style="padding: 10px; color: #16a34a; font-weight: bold;">{row.get('long_prob', 0)}%</td>
            <td style="padding: 10px; color: #dc2626; font-weight: bold;">{row.get('short_prob', 0)}%</td>
            <td style="padding: 10px;">{row.get('rsi', 50)}</td>
            <td style="padding: 10px;">{row.get('atr_pips', 0)} pips</td>
        </tr>
        """

    return f"""
    <div style="font-family: sans-serif; max-width: 650px; margin: auto; border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px;">
        <h2 style="color: #1e3a8a;">📈 GMO / SBI FX クオンツAI 到達確率通知</h2>
        <p style="color: #6b7280; font-size: 13px;">計測日時: {now_str}</p>
        <p>送信先: huashenfo@gmail.com</p>
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
            <thead>
                <tr style="background-color: #1e293b; color: white;">
                    <th style="padding: 10px;">通貨ペア</th>
                    <th style="padding: 10px;">現在値</th>
                    <th style="padding: 10px;">買い確率</th>
                    <th style="padd
