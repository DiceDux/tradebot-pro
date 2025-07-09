import pymysql

# مقادیر اتصال به دیتابیس را تنظیم کن
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',  # رمز دیتابیس
    database='tradebot-pro',
    charset='utf8mb4',
    autocommit=True
)

symbol_map = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'DOGE': 'DOGEUSDT',
    'SOL': 'SOLUSDT'
}

try:
    with conn.cursor() as cursor:
        # مرحله ۱: پیدا کردن رکوردهای تکراری بعد از تبدیل
        print("Finding duplicates...")
        find_dupes = """
        SELECT
            LEAST(n1.id, n2.id) AS id_to_delete
        FROM news n1
        JOIN news n2
          ON n1.published_at = n2.published_at
         AND (
            (n1.symbol IN %s AND n2.symbol IN %s AND
            (CASE n1.symbol
                WHEN 'BTC' THEN 'BTCUSDT'
                WHEN 'ETH' THEN 'ETHUSDT'
                WHEN 'DOGE' THEN 'DOGEUSDT'
                WHEN 'SOL' THEN 'SOLUSDT'
                ELSE n1.symbol END) = n2.symbol
            )
        )
        WHERE n1.id != n2.id
        """
        cursor.execute(find_dupes, (tuple(symbol_map.keys()), tuple(symbol_map.values())))
        ids = [row[0] for row in cursor.fetchall()]
        print(f"{len(ids)} duplicate rows to delete.")
        if ids:
            chunk_size = 200
            for i in range(0, len(ids), chunk_size):
                del_ids = ids[i:i+chunk_size]
                del_sql = f"DELETE FROM news WHERE id IN ({','.join(['%s']*len(del_ids))})"
                cursor.execute(del_sql, del_ids)
                print(f"Deleted {len(del_ids)} rows")

        # مرحله ۲: آپدیت نمادها به شکل درست
        print("Updating symbols...")
        update_sql = """
        UPDATE news
        SET symbol = CASE symbol
            WHEN 'BTC' THEN 'BTCUSDT'
            WHEN 'ETH' THEN 'ETHUSDT'
            WHEN 'DOGE' THEN 'DOGEUSDT'
            WHEN 'SOL' THEN 'SOLUSDT'
            ELSE symbol
        END
        WHERE symbol IN ('BTC', 'ETH', 'DOGE', 'SOL')
        """
        cursor.execute(update_sql)
        print("Update finished.")

finally:
    conn.close()
    print("Connection closed.")