import os
import sys
import datetime
import logging
import traceback
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# システムログ設定 & エラートレースバック用ヘルパー
# -----------------------------------------------------------------------------
LOG_FILE = "system_app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log_info(msg: str):
    logging.info(msg)
    st.session_state["logs"].append(f"[INFO {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def log_error(msg: str, exc: Exception = None):
    err_detail = f"{msg} | Exception: {exc}"
    if exc:
        err_detail += f"\n{traceback.format_exc()}"
    logging.error(err_detail)
    st.session_state["logs"].append(f"[ERROR {datetime.datetime.now().strftime('%H:%M:%S')}] {err_detail}")

# -----------------------------------------------------------------------------
# 定数 & Built to Last 企業データベース
# -----------------------------------------------------------------------------
PREDICTIONS_CSV = "predictions_log.csv"
APP_PASSWORD = "127812"

COUNTRY_TICKERS = {
    "🇺🇸 アメリカ": {
        "candidates": ["SPY", "QQQ", "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AMD"],
        "built_to_last": ["MSFT", "AAPL", "BRK-B"]
    },
    "🇯🇵 日本": {
        "candidates": ["1321.T", "1570.T", "7203.T", "6758.T", "6861.T", "8035.T", "9984.T", "6501.T", "8306.T"],
        "built_to_last": ["7203.T", "6758.T", "6861.T"]
    },
    "🇨🇳 中国": {
        "candidates": ["2833.HK", "0700.HK", "9988.HK", "1211.HK", "3690.HK", "9888.HK", "1810.HK", "600519.SS"],
        "built_to_last": ["0700.HK", "600519.SS", "1211.HK"]
    }
}

BUILT_TO_LAST_DATA = {
    "MSFT": {
        "name": "Microsoft Corporation",
        "symbol": "MSFT",
        "country": "🇺🇸 アメリカ",
        "sector": "テクノロジー / クラウド・AI",
        "moat": "Wide (極めて強固)",
        "per": "34.2",
        "roe": "38.5%",
        "net_margin": "36.2%",
        "debt_equity": "0.42",
        "moat_desc": "Windows / Officeの圧倒的シェア、Azureクラウドインフラ、OpenAI提携によるAIプラットフォーム標準化のネットワーク効果。",
        "rationale": "B2Bソフトウェアの不可欠な基盤であり、高いストック収益と圧倒的な自己資金創出力を誇るビジョナリー・カンパニーの筆頭。"
    },
    "AAPL": {
        "name": "Apple Inc.",
        "symbol": "AAPL",
        "country": "🇺🇸 アメリカ",
        "sector": "テクノロジー / コンシューマーハード・エコシステム",
        "moat": "Wide (極めて強固)",
        "per": "31.5",
        "roe": "145.0%",
        "net_margin": "26.1%",
        "debt_equity": "1.40",
        "moat_desc": "iPhoneを核とするエコシステムと高い顧客スイッチングコスト。サービス部門の継続課金モデルの強靭さ。",
        "rationale": "世界最高峰のブランド価値とエコシステム顧客囲い込みにより、経済変動に強い安定成長と強力な自社株買いを継続。"
    },
    "BRK-B": {
        "name": "Berkshire Hathaway Inc.",
        "symbol": "BRK-B",
        "country": "🇺🇸 アメリカ",
        "sector": "金融・保険・複合企業",
        "moat": "Wide (極めて強固)",
        "per": "19.8",
        "roe": "14.2%",
        "net_margin": "18.5%",
        "debt_equity": "0.22",
        "moat_desc": "保険事業（フロート資金）を活用した複利運用、分散された優良実業子会社群（鉄道、エネルギー、製造）。",
        "rationale": "ウォーレン・バフェットが築いた究極の「Built to Last」要塞。膨大な現金保有と強固な財務体質で不況期に真価を発揮。"
    },
    "7203.T": {
        "name": "トヨタ自動車",
        "symbol": "7203.T",
        "country": "🇯🇵 日本",
        "sector": "自動車・モビリティ",
        "moat": "Wide (強固)",
        "per": "9.8",
        "roe": "14.1%",
        "net_margin": "10.2%",
        "debt_equity": "0.58",
        "moat_desc": "TPS（トヨタ生産方式）による圧巻のコスト競争力、HV（ハイブリッド）の世界的覇権、次世代SDVへの大規模投資。",
        "rationale": "世界トップの販売台数と圧倒的な現地サプライチェーン網を誇る。ハイブリッド需要の再評価により長期的な収益性が安定。"
    },
    "6758.T": {
        "name": "ソニーグループ",
        "symbol": "6758.T",
        "country": "🇯🇵 日本",
        "sector": "エンターテインメント・電子部品",
        "moat": "Wide (強固)",
        "per": "17.4",
        "roe": "13.8%",
        "net_margin": "9.8%",
        "debt_equity": "0.45",
        "moat_desc": "PlayStationエコシステム、音楽・映画ライブラリIP、世界トップシェアのCMOSイメージセンサー技術。",
        "rationale": "ハードウェアからコンテンツIP・リカーリング型エンタメ企業へと見事に進化を遂げた日本を代表するグローバル企業。"
    },
    "6861.T": {
        "name": "キーエンス",
        "symbol": "6861.T",
        "country": "🇯🇵 日本",
        "sector": "FAセンサー・計測機器",
        "moat": "Wide (極めて強固)",
        "per": "42.1",
        "roe": "12.5%",
        "net_margin": "54.3%",
        "debt_equity": "0.00",
        "moat_desc": "直販営業による高付加価値提案（ファブレス経営）、圧倒的営業利益率50%超、自己資本比率90%以上の無借金経営。",
        "rationale": "世界の工場自動化（FA）における不可欠な存在。極めて高い利益率とコンサルティング型営業力で持続的成長を実現。"
    },
    "0700.HK": {
        "name": "Tencent Holdings (騰訊)",
        "symbol": "0700.HK",
        "country": "🇨🇳 中国",
        "sector": "インターネット・ゲーム・決済",
        "moat": "Wide (極めて強固)",
        "per": "22.5",
        "roe": "18.6%",
        "net_margin": "27.4%",
        "debt_equity": "0.31",
        "moat_desc": "中国国民的インフラ「WeChat (微信)」の巨大ネットワーク効果、世界最大級のオンラインゲームポートフォリオ。",
        "rationale": "13億人を超えるWeChat経済圏を背景に、SNS、クラウド、AI、フィンテック全方位で収益を生み出す中国デジタル経済の要石。"
    },
    "600519.SS": {
        "name": "Kweichow Moutai (貴州茅台)",
        "symbol": "600519.SS",
        "country": "🇨🇳 中国",
        "sector": "生活必需品 / 高級白酒",
        "moat": "Wide (極めて強固)",
        "per": "26.3",
        "roe": "32.1%",
        "net_margin": "52.8%",
        "debt_equity": "0.00",
        "moat_desc": "中国最高の国酒ブランド価値、地理的表示による唯一無二の希少性、営業利益率65%超の圧倒的価格決定力。",
        "rationale": "中国文化に深く根ざしたラグジュアリー商品。強力な価格決定力と圧倒的な純利益率を保持し、株主還元も極めて強固。"
    },
    "1211.HK": {
        "name": "BYD Company (比亜迪)",
        "symbol": "1211.HK",
        "country": "🇨🇳 中国",
        "sector": "EV・車載バッテリー",
        "moat": "Wide (強固)",
        "per": "19.2",
        "roe": "21.5%",
        "net_margin": "5.6%",
        "debt_equity": "0.62",
        "moat_desc": "バッテリーから半導体・車体まで手掛ける垂直統合モデル、大規模製造による圧倒的EVコスト破壊力。",
        "rationale": "世界最大規模のNEV（新エネルギー車）メーカー。完全自社内製化によるコスト競争力を武器に、グローバル市場へ急拡大。"
    }
}

# -----------------------------------------------------------------------------
# 認証機能 (パスワード確認)
# -----------------------------------------------------------------------------
def check_password():
    """パスワード認証のUIおよびセッション管理を行う関数"""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 パスワード保護されたアプリケーション")
        input_password = st.text_input("パスワードを入力してください:", type="password")
        
        if st.button("ログイン"):
            if input_password == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("パスワードが正しくありません。")
        return False
    return True

# -----------------------------------------------------------------------------
# 予測ログ管理 (Prediction Tracker) & 実績評価機能
# -----------------------------------------------------------------------------
def init_predictions_log():
    if not os.path.exists(PREDICTIONS_CSV):
        df = pd.DataFrame(columns=[
            "datetime", "country", "ticker", "target_period",
            "direction", "initial_price", "target_price",
            "prob", "status", "actual_return", "outcome"
        ])
        now = datetime.datetime.now()
        sample_data = [
            [(now - datetime.timedelta(days=90)).strftime("%Y-%m-%d %H:%M"), "🇺🇸 アメリカ", "NVDA", "3ヶ月", "Long", 450.0, 520.0, 0.78, "Completed", 22.5, 1],
            [(now - datetime.timedelta(days=80)).strftime("%Y-%m-%d %H:%M"), "🇯🇵 日本", "7203.T", "3ヶ月", "Long", 2400.0, 2600.0, 0.65, "Completed", 8.3, 1],
            [(now - datetime.timedelta(days=70)).strftime("%Y-%m-%d %H:%M"), "🇨🇳 中国", "9988.HK", "3ヶ月", "Long", 85.0, 95.0, 0.58, "Completed", -4.2, 0],
            [(now - datetime.timedelta(days=60)).strftime("%Y-%m-%d %H:%M"), "🇺🇸 アメリカ", "MSFT", "3ヶ月", "Long", 380.0, 410.0, 0.72, "Completed", 10.5, 1],
            [(now - datetime.timedelta(days=50)).strftime("%Y-%m-%d %H:%M"), "🇯🇵 日本", "6861.T", "3ヶ月", "Long", 62000.0, 68000.0, 0.69, "Completed", 6.8, 1],
            [(now - datetime.timedelta(days=40)).strftime("%Y-%m-%d %H:%M"), "🇺🇸 アメリカ", "QQQ", "3ヶ月", "Long", 410.0, 440.0, 0.75, "Completed", 7.2, 1],
            [(now - datetime.timedelta(days=30)).strftime("%Y-%m-%d %H:%M"), "🇨🇳 中国", "0700.HK", "3ヶ月", "Long", 320.0, 360.0, 0.61, "Completed", 12.1, 1],
            [(now - datetime.timedelta(days=20)).strftime("%Y-%m-%d %H:%M"), "🇯🇵 日本", "8035.T", "3ヶ月", "Long", 25000.0, 27000.0, 0.64, "Completed", -2.1, 0],
            [(now - datetime.timedelta(days=15)).strftime("%Y-%m-%d %H:%M"), "🇺🇸 アメリカ", "AAPL", "3ヶ月", "Long", 180.0, 195.0, 0.68, "Completed", 5.4, 1],
            [(now - datetime.timedelta(days=10)).strftime("%Y-%m-%d %H:%M"), "🇯🇵 日本", "1321.T", "3ヶ月", "Long", 38000.0, 40000.0, 0.71, "Completed", 3.2, 1],
        ]
        df_sample = pd.DataFrame(sample_data, columns=df.columns)
        df_sample.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
        log_info("Initialized predictions_log.csv with baseline history.")

def save_prediction_log(country, ticker, target_period, direction, initial_price, target_price, prob):
    try:
        init_predictions_log()
        new_row = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "country": country,
            "ticker": ticker,
            "target_period": target_period,
            "direction": direction,
            "initial_price": float(initial_price),
            "target_price": float(target_price),
            "prob": float(prob),
            "status": "Pending",
            "actual_return": 0.0,
            "outcome": 0
        }
        df = pd.read_csv(PREDICTIONS_CSV, encoding="utf-8-sig")
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
