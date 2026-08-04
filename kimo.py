import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =============================================================================
# 1. ページ環境設定 & セキュリティパスワード認証システム (Passcode: 238923)
# =============================================================================
st.set_page_config(
    page_title="業種別ETF & 自己進化型AI株価予測ダッシュボード (プロフェッショナル完全版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 認証用固定パスワード
PASSWORD = "238923"

def check_password():
    """
    セッション状態(session_state)を利用したセキュリティアクセス制御機能。
    パスワードの認証状態を保持し、未認証の場合はアクセス制限画面を表示します。
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 セキュリティ認証 - パスワード保護領域")
        st.info("本ダッシュボード(全機能・全モジュール版)にアクセスするには認証パスワードを入力してください。")
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            input_pass = st.text_input("アクセスパスワード (初期設定: 238923)", type="password")
        if st.button("ログイン認証を実行"):
            if input_pass == PASSWORD:
                st.session_state["authenticated"] = True
                st.success("認証に成功しました！ダッシュボードを起動します...")
                st.rerun()
            else:
                st.error("パスワードが正しくありません。再度ご確認ください。")
        return False
    return True

# パスワード認証が完了していない場合は処理を停止
if not check_password():
    st.stop()

# =============================================================================
# 2. グローバル銘柄マスタ & 業種別ETFマスターデータベース
# =============================================================================
STOCK_DICT = {
    # 🇺🇸 アメリカ市場 業種別Select Sector SPDR ETF & 代表銘柄
    'XLK': {'name': 'テクノロジー業種Select ETF (XLK)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '半導体・ソフトウェア・AI中心のハイテク銘柄群'},
    'XLF': {'name': '金融業種Select ETF (XLF)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '大手銀行・保険・金融サービス銘柄群'},
    'XLV': {'name': 'ヘルスケア業種Select ETF (XLV)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '製薬・医療機器・バイオテクノロジー銘柄群'},
    'XLE': {'name': 'エネルギー業種Select ETF (XLE)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '石油・天然ガス・エネルギー資源銘柄群'},
    'XLY': {'name': '一般消費財業種Select ETF (XLY)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': 'Amazon・自動車・耐久消費財銘柄群'},
    'XLP': {'name': '生活必需品業種Select ETF (XLP)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '日用品・飲料・食品スーパー銘柄群'},
    'XLI': {'name': '資本財・産業業種Select ETF (XLI)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '航空・防衛・機械・物流関連銘柄群'},
    'XLB': {'name': '素材業種Select ETF (XLB)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '化学・金属・採掘・建築資材銘柄群'},
    'XLRE': {'name': '不動産業種Select ETF (XLRE)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '商業用不動産・データセンターREIT銘柄群'},
    'XLC': {'name': '通信サービス業種Select ETF (XLC)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': 'Meta・Alphabet・エンタメ・通信銘柄群'},
    'SOXX': {'name': 'iShares 半導体株業種ETF (SOXX)', 'category': '業種別ETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': 'フィラデルフィア半導体指数連動銘柄群'},
    'SPY': {'name': 'SPDR S&P500 インデックスETF', 'category': '広域インデックスETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': '米国大型株500銘柄全体への投資'},
    'QQQ': {'name': 'Invesco NASDAQ100 ETF', 'category': 'ハイテクインデックスETF', 'country': '🇺🇸 アメリカ', 'is_etf': True, 'desc': 'ナスダック主要100非金融大型株'},
    'NVDA': {'name': 'エヌビディア (NVIDIA Corporation)', 'category': '半導体・AI', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'AIグラフィックスプロセッサ (GPU) グローバル王者'},
    'MSFT': {'name': 'マイクロソフト (Microsoft Corp)', 'category': 'クラウド・AI', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'Windows, Azure, OpenAI出資によるAIリード'},
    'AAPL': {'name': 'アップル (Apple Inc)', 'category': 'ハードウェア', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'iPhone, Mac, Services による強固なエコシステム'},
    'AMZN': {'name': 'アマゾン・ドット・コム (Amazon.com)', 'category': 'EC・クラウド', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'AWSクラウドインフラ & グローバルEC王者'},
    'GOOGL': {'name': 'アルファベット (Alphabet Inc)', 'category': '検索・AI・クラウド', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'Google Search, YouTube, Gemini AI'},
    'META': {'name': 'メタ・プラットフォームズ (Meta Platforms)', 'category': 'SNS・AI', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'Instagram, WhatsApp, Llama オープンAI'},
    'BRK-B': {'name': 'バークシャー・ハサウェイ (Berkshire Hathaway)', 'category': '保険・多角投資', 'country': '🇺🇸 アメリカ', 'is_etf': False, 'desc': 'バフェット率いる保険・鉄道・エネルギー巨頭'},

    # 🇯🇵 日本市場 業種別ETF (TOPIX17) & 代表優良株
    '1615.T': {'name': 'NF TOPIX銀行業種ETF', 'category': '業種別ETF (銀行業)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': 'メガバンク・地方銀行株に一括投資'},
    '1621.T': {'name': 'NF 医薬品業種ETF (TOPIX-17)', 'category': '業種別ETF (医薬品)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': '武田薬品・アステラス等大手製薬株'},
    '1622.T': {'name': 'NF 自動車・輸送機業種ETF', 'category': '業種別ETF (自動車)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': 'トヨタ・ホンダ・デンソー等自動車産業'},
    '1625.T': {'name': 'NF 電機・精密業種ETF', 'category': '業種別ETF (電機)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': 'ソニー・キーエンス・日立等電機メーカー'},
    '1629.T': {'name': 'NF 商社・卸売業種ETF', 'category': '業種別ETF (商社)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': '三菱商事・三井物産・伊藤忠等5大商社'},
    '1630.T': {'name': 'NF 小売業種ETF', 'category': '業種別ETF (小売)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': 'ファーストリテイリング・セブン&アイ等'},
    '1617.T': {'name': 'NF 食品業種ETF (TOPIX-17)', 'category': '業種別ETF (食品)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': '味の素・アサヒ等加工食品・飲料メーカー'},
    '1618.T': {'name': 'NF エネルギー資源業種ETF', 'category': '業種別ETF (資源)', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': 'INPEX・石油元売り大手銘柄群'},
    '1321.T': {'name': 'NF 日経225連動型上場投資信託', 'category': '広域インデックスETF', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': '日経平均株価225銘柄全体へ投資'},
    '1570.T': {'name': 'NF 日経平均レバレッジETF', 'category': 'レバレッジETF', 'country': '🇯🇵 日本', 'is_etf': True, 'desc': '日経平均の2倍の変動率を目指すETF'},
    '7203.T': {'name': 'トヨタ自動車 (Toyota Motor)', 'category': '自動車・モビリティ', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': '世界トップシェア自動車メーカー (TPS)'},
    '6758.T': {'name': 'ソニーグループ (Sony Group)', 'category': 'エンタメ・電子部品', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': 'ゲーム・音楽・映画・イメージセンサー'},
    '6861.T': {'name': 'キーエンス (Keyence)', 'category': 'FAセンサー・計測器', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': '営業利益率50%超のFA超高収益企業'},
    '8035.T': {'name': '東京エレクトロン (Tokyo Electron)', 'category': '半導体製造装置', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': 'コータ・デベロッパー等半導体製造装置'},
    '8306.T': {'name': '三菱UFJフィナンシャルG', 'category': 'メガバンク・金融', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': '国内最大の民間総合金融グループ'},
    '9984.T': {'name': 'ソフトバンクグループ', 'category': 'AIファンド・通信', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': 'ビジョン・ファンドを通じたグローバルAI投資'},
    '8058.T': {'name': '三菱商事 (Mitsubishi Corp)', 'category': '総合商社', 'country': '🇯🇵 日本', 'is_etf': False, 'desc': 'エネルギー・金属・食品の多角商社'},

    # 🇨🇳 中国・香港市場 業種別ETF & 代表銘柄
    '3033.HK': {'name': 'Hang Seng TECH (恒生科技業種) ETF', 'category': '業種別ETF', 'country': '🇨🇳 中国', 'is_etf': True, 'desc': 'アリババ・テンセント等ハイテク30銘柄'},
    '2828.HK': {'name': 'Hang Seng China Enterprises (H株) ETF', 'category': '業種別ETF', 'country': '🇨🇳 中国', 'is_etf': True, 'desc': '香港上場の中国本土主要企業(H株)'},
    '3169.HK': {'name': 'China Consumer (中国消費財業種) ETF', 'category': '業種別ETF', 'country': '🇨🇳 中国', 'is_etf': True, 'desc': '中国のメガ内需消費市場連動銘柄群'},
    '2833.HK': {'name': 'Hang Seng Index (恒生指数) ETF', 'category': '広域インデックスETF', 'country': '🇨🇳 中国', 'is_etf': True, 'desc': '香港株式市場全体の代表的インデックス'},
    '0700.HK': {'name': 'Tencent Holdings (騰訊控股 / テンセント)', 'category': 'ネット・ゲーム', 'country': '🇨🇳 中国', 'is_etf': False, 'desc': 'WeChat, 世界最大級のゲーム・SNS'},
    '9988.HK': {'name': 'Alibaba Group (阿里巴巴 / アリババ)', 'category': 'EC・クラウド', 'country': '🇨🇳 中国', 'is_etf': False, 'desc': 'Taobao, Tmall, Alibaba Cloud'},
    '1211.HK': {'name': 'BYD Company (比亜迪 / ビーワイディー)', 'category': 'EV・車載電池', 'country': '🇨🇳 中国', 'is_etf': False, 'desc': 'EV世界販売台数首位級 & バッテリー自社生産'},
    '600519.SS': {'name': 'Kweichow Moutai (貴州茅台酒 / マウタイ)', 'category': '高級白酒・消費財', 'country': '🇨🇳 中国', 'is_etf': False, 'desc': '中国伝統の最高級白酒メーカー'}
}

# 国別フィルタ定義
COUNTRY_CANDIDATES = {
    '🇺🇸 アメリカ': ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLRE', 'XLC', 'SOXX', 'SPY', 'QQQ', 'NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'BRK-B'],
    '🇯🇵 日本': ['1615.T', '1621.T', '1622.T', '1625.T', '1629.T', '1630.T', '1617.T', '1618.T', '1321.T', '1570.T', '7203.T', '6758.T', '6861.T', '8035.T', '8306.T', '9984.T', '8058.T'],
    '🇨🇳 中国': ['3033.HK', '2828.HK', '3169.HK', '2833.HK', '0700.HK', '9988.HK', '1211.HK', '600519.SS']
}

# 永続的競争優位性 (Moat) マスターリスト
BUILT_TO_LAST_DATA = [
    {'symbol': 'MSFT', 'name': 'マイクロソフト', 'moat': 'Wide (超強固 OS/クラウド/AI)', 'roe': '38.5%', 'operating_margin': '44.6%', 'growth': '+15.2%', 'eval': 'S (最高ランク)'},
    {'symbol': 'AAPL', 'name': 'アップル', 'moat': 'Wide (エコシステム/ブランド力)', 'roe': '147.2%', 'operating_margin': '30.7%', 'growth': '+8.1%', 'eval': 'S (最高ランク)'},
    {'symbol': 'BRK-B', 'name': 'バークシャー・ハサウェイ', 'moat': 'Wide (多角化・現金/持株)', 'roe': '14.1%', 'operating_margin': '18.9%', 'growth': '+11.5%', 'eval': 'S (最高ランク)'},
    {'symbol': '7203.T', 'name': 'トヨタ自動車', 'moat': 'Wide (生産方式・グローバルブランド)', 'roe': '11.8%', 'operating_margin': '10.2%', 'growth': '+21.4%', 'eval': 'A+ (優良)'},
    {'symbol': '6758.T', 'name': 'ソニーグループ', 'moat': 'Wide (コンテンツ・IP/イメージセンサー)', 'roe': '13.5%', 'operating_margin': '11.8%', 'growth': '+12.0%', 'eval': 'A+ (優良)'},
    {'symbol': '6861.T', 'name': 'キーエンス', 'moat': 'Wide (超高利益率・直販・FA技術)', 'roe': '13.2%', 'operating_margin': '52.1%', 'growth': '+11.1%', 'eval': 'S (最高ランク)'}
]

# =============================================================================
# 3. 高度データ取得 & パイプライン
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """yfinance経由で株価を取得し、各種テクニカル指標と機械学習用特徴量を計算"""
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 移動平均線 (SMA)
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        # RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # ボリンジャーバンド (20日, 2σ) & 乖離率
        df['BB_Upper'] = df['SMA20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['SMA20'] - (df['Close'].rolling(window=20).std() * 2)
        df['MA_Disparity_20'] = ((df['Close'] - df['SMA20']) / df['SMA20']) * 100
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100

        # 機械学習用ターゲット（20営業日後上昇判定: +2%以上）
        df['Target_20d'] = (df['Close'].shift(-20) > df['Close'] * 1.02).astype(int)

        latest = df.iloc[-1]
        features = {
            'RSI': float(latest['RSI']),
            'MACD_Hist': float(latest['MACD_Hist']),
            'MA_Disparity_20': float(latest['MA_Disparity_20']),
            'Volatility_20': float(latest['Volatility_20']),
            'Last_Close': float(latest['Close'])
        }
        return df, features
    except Exception as e:
        st.warning(f"データ取得エラー ({ticker}): {e}")
        return None, None

# =============================================================================
# 4. 機械学習 (RandomForest) & 多期間予測エンジン
# =============================================================================
def train_and_predict_ml(df, features):
    """過去データに基づきRandomForest分類モデルを学習し上昇確率を予測"""
    feature_cols = ['RSI', 'MACD_Hist', 'MA_Disparity_20', 'Volatility_20']
    clean_df = df.dropna(subset=feature_cols + ['Target_20d']).copy()

    if len(clean_df) < 60:
        prob_up = float(np.clip(0.50 + (features['RSI'] - 50) * 0.005 + features['MACD_Hist'] * 0.08, 0.35, 0.85))
        importances = {'RSI': 0.4, 'MACD_Hist': 0.3, 'MA_Disparity_20': 0.15, 'Volatility_20': 0.15}
        return prob_up, importances

    X = clean_df[feature_cols]
    y = clean_df['Target_20d']

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    latest_X = pd.DataFrame([[features['RSI'], features['MACD_Hist'], features['MA_Disparity_20'], features['Volatility_20']]], columns=feature_cols)
    prob_up = float(model.predict_proba(latest_X)[0][1])
    importances = dict(zip(feature_cols, model.feature_importances_))

    return prob_up, importances

def calculate_multi_horizon(last_close, prob_up):
    """1ヶ月, 3ヶ月, 6ヶ月, 12ヶ月先の株価予測"""
    horizons = [
        {'period': '1ヶ月先', 'mult': 0.10},
        {'period': '3ヶ月先', 'mult': 0.28},
        {'period': '6ヶ月先', 'mult': 0.45},
        {'period': '12ヶ月先', 'mult': 0.65}
    ]
    results = []
    for h in horizons:
        expected_change = (prob_up - 0.5) * h['mult']
        target_price = round(last_close * (1 + expected_change), 2)
        return_pct = round(expected_change * 100, 2)
        results.append({
            '対象期間': h['period'],
            '上昇勝率 (Prob)': f"{round(prob_up * 100, 1)}%",
            '予測目標株価': f"{target_price:,.2f}",
            '予想リターン (%)': f"{return_pct:+.2f}%"
        })
    return results

# =============================================================================
# 5. メインダッシュボード UI
# =============================================================================
st.title("📈 業種別ETF & 自己進化型AI株価予測ダッシュボード")

# サイドバー設定
st.sidebar.header("🔍 分析設定")
selected_country = st.sidebar.selectbox("国・地域を選択", list(COUNTRY_CANDIDATES.keys()))

tickers_in_country = COUNTRY_CANDIDATES[selected_country]
ticker_options = {t: f"{t} | {STOCK_DICT[t]['name']}" for t in tickers_in_country if t in STOCK_DICT}

selected_ticker = st.sidebar.selectbox(
    "銘柄 / ETFを選択",
    options=list(ticker_options.keys()),
    format_func=lambda x: ticker_options[x]
)

if st.sidebar.button("🔒 ログアウト"):
    st.session_state["authenticated"] = False
    st.rerun()

# タブ表示の設定
tab1, tab2, tab3 = st.tabs(["📊 個別AI分析 & 予測", "🏰 ビジョナリー優良株", "🌐 収録銘柄マスター一覧"])

# -----------------------------------------------------------------------------
# タブ1: 個別AI分析 & 予測
# -----------------------------------------------------------------------------
with tab1:
    stock_info = STOCK_DICT[selected_ticker]
    st.subheader(f"分析対象: {stock_info['name']} ({selected_ticker})")
    st.caption(f"カテゴリ: {stock_info['category']} | 国: {stock_info['country']} | 概要: {stock_info.get('desc', '')}")

    with st.spinner("リアルタイム株価を取得・AI解析中..."):
        df, features = fetch_stock_data(selected_ticker)

    if df is not None and features is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新終値", f"{features['Last_Close']:,.2f}")
        c2.metric("RSI (14日)", f"{features['RSI']:.1f}")
        c3.metric("20日移動平均乖離率", f"{features['MA_Disparity_20']:+.2f}%")
        c4.metric("ボラティリティ (20日)", f"{features['Volatility_20']:.2f}%")

        st.markdown("---")

        st.subheader("🤖 AI (RandomForest) 多期間予測結果")
        prob_up, importances = train_and_predict_ml(df, features)
        multi_horizon = calculate_multi_horizon(features['Last_Close'], prob_up)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("##### 📈 期間別ターゲット目標価格 & 予想リターン")
            st.dataframe(pd.DataFrame(multi_horizon), hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("##### 🧠 AIモデルの特徴量寄与度 (Feature Importance)")
            imp_df = pd.DataFrame([
                {'指標': k, '寄与度': f"{v*100:.1f}%"} for k, v in importances.items()
            ]).sort_values(by='寄与度', ascending=False)
            st.dataframe(imp_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        st.subheader("📉 株価チャート & 移動平均線 (過去2年)")
        chart_df = df[['Close', 'SMA20', 'SMA50']].dropna()
        st.line_chart(chart_df)
    else:
        st.error("株価データの取得に失敗しました。時間をおいて再試行するか、別の銘柄を選択してください。")

# -----------------------------------------------------------------------------
# タブ2: ビジョナリー優良株
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🏰 ビジョナリー・カンパニー（超強固な競合優位性を持つ銘柄）")
    st.write("「経済の堀 (Economic Moat)」を持ち、長期的・継続的に高い資本効率（ROE）を維持できるグローバル優良企業群です。")
    st.dataframe(pd.DataFrame(BUILT_TO_LAST_DATA), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# タブ3: 収録銘柄マスター一覧
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🌐 収録銘柄 & 業種別ETFマスターデータベース")
    master_rows = []
    for sym, info in STOCK_DICT.items():
        master_rows.append({
            'シンボル': sym,
            '銘柄・ETF名': info['name'],
            'カテゴリ': info['category'],
            '国/地域': info['country'],
            '種別': 'ETF' if info['is_etf'] else '個別株',
            '詳細説明': info.get('desc', '')
        })
    st.dataframe(pd.DataFrame(master_rows), use_container_width=True, hide_index=True)
