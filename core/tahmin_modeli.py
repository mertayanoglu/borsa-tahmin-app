import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import time
import json
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

# Sayfa yapılandırması
st.set_page_config(page_title="BIST Tahmin Robotu", layout="wide")
st_autorefresh(interval=60 * 1000, key="auto-refresh")

BIST_40 = [
    "AKBNK", "AKSEN", "ALARK", "ASELS", "BIMAS", "DOHOL", "EKGYO", "ENJSA", "EREGL", "FROTO",
    "GARAN", "GUBRF", "HALKB", "HEKTS", "ISCTR", "KCHOL", "KOZAA", "KOZAL", "KRDMD", "MGROS",
    "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO",
    "TSKB", "TTKOM", "TTRAK", "TUPRS", "VAKBN", "VESBE", "YKBNK", "SOKM", "SKBNK", "ARCLK"
]

def get_hisse_verisi(symbol="AKBNK", gun=90):
    symbol_yf = symbol + ".IS"
    df = yf.download(symbol_yf, period=f"{gun}d", interval="1d", progress=False)
    if df is None or len(df) < 30:
        raise ValueError(f"Yetersiz veri ({len(df) if df is not None else 0} satır)")
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    df["symbol"] = symbol

    df["EMA_10"] = df["close"].ewm(span=10).mean()
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    df["volume_change"] = df["Volume"].pct_change()
    df["prev_return"] = df["close"].pct_change()
    df["price_diff_3day"] = df["close"] - df["close"].shift(3)
    df["price_volatility"] = df["close"].rolling(window=5).std()

    df.dropna(inplace=True)
    df["y_price_increase"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

def tahmin_uret(symbol):
    try:
        df = get_hisse_verisi(symbol)
        features = ["EMA_10", "EMA_20", "RSI_14", "MACD", "volume_change", "prev_return", "price_diff_3day", "price_volatility"]
        X = df[features]
        y = df["y_price_increase"]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        model = HistGradientBoostingClassifier(max_iter=200, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        latest_features = scaler.transform([df[features].iloc[-1].values])
        prediction = model.predict(latest_features)[0]

        open_price = float(df["close"].iloc[-1])
        current = yf.Ticker(symbol + ".IS").history(period="1d")
        change_pct = None
        current_price = None

        if not current.empty:
            current_price = float(current["Close"].iloc[-1])
            change_pct = ((current_price - open_price) / open_price) * 100

        target_price = round(open_price * 1.03, 2) if prediction else round(open_price * 0.97, 2)

        return {
            "Hisse": symbol,
            "Model Doğruluğu": round(acc, 2),
            "Son Kapanış": open_price,
            "Tahmin": "⬆️ Artabilir" if prediction else "⬇️ Düşebilir",
            "Gerçek Durum (%)": round(change_pct, 2) if change_pct is not None else None,
            "Fiyat Farkı": round(current_price - open_price, 2) if current_price is not None else None,
            "Hedef Fiyat": target_price,
            "Güncel Fiyat": current_price
        }

    except Exception as e:
        return {
            "Hisse": symbol,
            "Model Doğruluğu": None,
            "Son Kapanış": None,
            "Tahmin": f"Hata: {str(e)}",
            "Gerçek Durum (%)": None,
            "Fiyat Farkı": None,
            "Hedef Fiyat": None,
            "Güncel Fiyat": None
        }

# Başlık
st.title("📊 BIST 40 Tahmin Robotu (yfinance)")
st.caption("Tahminler sadece butona tıklanınca güncellenir. Fiyatlar her dakikada otomatik güncellenir.")

# 1. Session state başlat
if "tahminler" not in st.session_state:
    if os.path.exists("tahmin_log.json"):
        with open("tahmin_log.json", "r") as f:
            st.session_state["tahminler"] = json.load(f)
    else:
        st.session_state["tahminler"] = []

# 2. Butona basılınca tahminleri güncelle
if st.button("🔁 Tahminleri Yeniden Hesapla"):
    sonuc = []
    progress = st.progress(0, text="Tahminler hesaplanıyor...")
    for i, symbol in enumerate(BIST_40):
        progress.progress((i + 1) / len(BIST_40))
        result = tahmin_uret(symbol)
        if result["Model Doğruluğu"] is not None and not str(result["Tahmin"]).startswith("Hata"):
            sonuc.append(result)
    progress.empty()
    st.session_state["tahminler"] = sonuc
    with open("tahmin_log.json", "w") as f:
        json.dump(sonuc, f)

# 3. Tahminleri güncel verilerle göster
if st.session_state["tahminler"]:
    df_sonuc = pd.DataFrame(st.session_state["tahminler"])

    # Güncel fiyatları yeniden al
    for i, row in df_sonuc.iterrows():
        try:
            current = yf.Ticker(row["Hisse"] + ".IS").history(period="1d")
            if not current.empty:
                current_price = float(current["Close"].iloc[-1])
                df_sonuc.at[i, "Güncel Fiyat"] = current_price
                df_sonuc.at[i, "Fiyat Farkı"] = round(current_price - row["Son Kapanış"], 2)
                df_sonuc.at[i, "Gerçek Durum (%)"] = round((current_price - row["Son Kapanış"]) / row["Son Kapanış"] * 100, 2)
        except:
            pass

    df_sonuc["Hedefe Ulaştı"] = df_sonuc.apply(lambda row: (
        row["Gerçek Durum (%)"] >= 3 and row["Tahmin"] == "⬆️ Artabilir") or
        (row["Gerçek Durum (%)"] <= -3 and row["Tahmin"] == "⬇️ Düşebilir"), axis=1)

    df_sonuc = df_sonuc.sort_values(by="Model Doğruluğu", ascending=False)

    basari_orani = round(df_sonuc["Hedefe Ulaştı"].mean() * 100, 2)
    st.success(f"✅ Tahmin Başarı Oranı: {basari_orani}%")
    st.subheader("🎯 Toplam Tahmin Sayısı: " + str(len(df_sonuc)))

    st.subheader("⬆️ Artması Beklenenler")
    st.dataframe(df_sonuc[df_sonuc["Tahmin"] == "⬆️ Artabilir"], use_container_width=True)

    st.subheader("⬇️ Düşmesi Beklenenler")
    st.dataframe(df_sonuc[df_sonuc["Tahmin"] == "⬇️ Düşebilir"], use_container_width=True)
else:
    st.info("📭 Henüz tahmin yapılmadı. Lütfen yukarıdaki butona tıklayın.")
