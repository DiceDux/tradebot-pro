import pymysql
import pandas as pd
from utils.config import DB_CONFIG, SYMBOLS

def analyze_news_gaps(symbol):
    conn = pymysql.connect(**DB_CONFIG)
    short_symbol = symbol.replace("USDT", "") if symbol.endswith("USDT") else symbol

    # اول سعی کن با symbol کامل بگیری، اگر نبود، با نام کوتاه بگیر
    for sym in [symbol, short_symbol]:
        query = """
            SELECT published_at
            FROM news
            WHERE symbol = %s
            ORDER BY published_at ASC
        """
        df = pd.read_sql(query, conn, params=(sym,))
        if not df.empty:
            print(f"\n[INFO] Found news for symbol: {sym} (original request: {symbol})")
            break

    conn.close()

    if df.empty:
        print(f"{symbol}: No news data found for symbol '{symbol}' or '{short_symbol}'.")
        return

    df['published_at'] = pd.to_datetime(df['published_at'])
    df = df.sort_values('published_at').reset_index(drop=True)

    # اختلاف زمانی بین هر دو خبر متوالی (برحسب ساعت)
    df['gap_hrs'] = df['published_at'].diff().dt.total_seconds() / 3600

    # بازه‌های بدون خبر (مثلاً بیشتر از 5 ساعت فاصله)
    gap_threshold = 5  # ساعت
    gap_idxs = df.index[df['gap_hrs'] > gap_threshold].tolist()

    print(f"\n--- {symbol} ---")
    print(f"Total news count: {len(df)}")
    print(f"First news:  {df['published_at'].iloc[0]}")
    print(f"Last news:   {df['published_at'].iloc[-1]}")
    print(f"Number of gaps > {gap_threshold}h: {len(gap_idxs)}")

    if gap_idxs:
        print("Gaps (prev_news_time --> next_news_time):")
        for idx in gap_idxs:
            prev_time = df['published_at'].iloc[idx-1]
            next_time = df['published_at'].iloc[idx]
            gap_hr = df['gap_hrs'].iloc[idx]
            print(f"  {prev_time}  -->  {next_time}   | gap: {gap_hr:.2f}h")
    else:
        print("No significant gaps found (all news are close together).")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        analyze_news_gaps(symbol)