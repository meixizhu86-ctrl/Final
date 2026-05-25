import gzip
import shutil
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(r"D:\FinTech\shioaji.db")
OUT_PATH = ROOT / "data" / "stock_KBar_2330_2022_2024.csv.gz"


def main() -> None:
    query = """
        select time, open, high, low, close, volume, amount, product
        from stock_KBar_2330
        where time between ? and ?
        order by time
    """
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(query, con, params=("2022-01-01", "2024-04-09 23:59:59"))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_PATH.with_suffix("")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(csv_path, "rb") as src, gzip.open(OUT_PATH, "wb") as dst:
        shutil.copyfileobj(src, dst)
    csv_path.unlink()
    print(f"exported {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
