import pandas as pd
import streamlit as st

from ai_report import answer_followup_question, build_ai_report, get_api_key_from_streamlit
from backtest_engine import load_price_data, metrics_frame, resample_kbars
from strategies import STRATEGIES, STRATEGY_DESCRIPTIONS, build_best_parameter_summary, buy_and_hold, optimize_strategy


DATA_PATH = "data/stock_KBar_2330_2022_2024.csv.gz"
METRIC_COLUMNS = [
    "total_profit",
    "trade_count",
    "win_rate",
    "avg_profit",
    "max_drawdown",
    "profit_factor",
    "return_drawdown_ratio",
    "sharpe_like",
    "expectancy",
    "score",
]


st.set_page_config(page_title="台股策略回測與 AI 評估", layout="wide")
st.title("台股策略回測與 AI 評估")
st.caption("資料來源：shioaji.db / stock_KBar_2330，策略包含 MA、RSI、布林通道、MACD、KDJ 與參數最佳化。")


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_price_data(DATA_PATH)


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.columns:
        if col == "win_rate":
            formatted[col] = formatted[col].map(lambda x: f"{float(x):.2%}")
        elif col in METRIC_COLUMNS and col != "trade_count":
            formatted[col] = formatted[col].map(lambda x: round(float(x), 4))
    return formatted


raw_df = load_data()

with st.sidebar:
    st.header("回測設定")
    min_date = raw_df["time"].min().date()
    max_date = raw_df["time"].max().date()
    date_range = st.date_input("回測日期", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    kbar_rule = st.selectbox("KBar 週期", ["1min", "5min", "15min", "30min", "60min", "1D"], index=2)
    quantity = st.number_input("每次交易數量", min_value=1, max_value=100, value=1, step=1)
    stop_loss = st.number_input("移動停損點數（0 表示不使用）", min_value=0.0, value=0.0, step=1.0)
    selected_strategies = st.multiselect("策略", list(STRATEGIES.keys()), default=list(STRATEGIES.keys()))
    run_optimization = st.checkbox("執行參數最佳化", value=False)
    temperature = st.slider("AI temperature", 0.0, 1.0, 0.3, 0.1)

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = raw_df[(raw_df["time"] >= start_date) & (raw_df["time"] < end_date)].copy()
else:
    filtered = raw_df.copy()

df = resample_kbars(filtered, kbar_rule)
if df.empty:
    st.error("目前日期區間沒有資料，請調整回測日期。")
    st.stop()

st.info(f"分析標的：2330 台積電；期間：{df['time'].min()} 至 {df['time'].max()}；K 棒數：{len(df):,}")

with st.expander("策略說明與績效評估方式", expanded=True):
    st.markdown(
        """
        本系統以技術指標產生買賣訊號，並將不同策略放在相同資料區間與交易條件下比較。
        績效評估除了觀察總損益，也納入勝率、最大回撤、報酬回撤比、類 Sharpe 與每筆交易期望值，
        讓策略比較同時兼顧報酬與風險。
        """
    )
    st.table(pd.DataFrame({"策略": list(STRATEGY_DESCRIPTIONS.keys()), "交易邏輯": list(STRATEGY_DESCRIPTIONS.values())}))

st.subheader("價格走勢")
st.line_chart(df.set_index("time")["close"], height=260)

base_params = {"quantity": int(quantity), "stop_loss": float(stop_loss)}
results = {strategy_name: STRATEGIES[strategy_name](df, **base_params) for strategy_name in selected_strategies}

if not results:
    st.warning("請至少選擇一個策略。")
    st.stop()

comparison = metrics_frame(results)
benchmark = buy_and_hold(df, quantity=int(quantity))
benchmark_row = pd.DataFrame([{"strategy": "買進持有基準", **benchmark.metrics}])
comparison_with_benchmark = pd.concat([comparison, benchmark_row], ignore_index=True)

st.subheader("策略績效比較")
st.dataframe(format_metrics(comparison_with_benchmark), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.download_button("下載績效比較 CSV", comparison_with_benchmark.to_csv(index=False, encoding="utf-8-sig"), "strategy_metrics.csv", "text/csv")
c2.download_button("下載價格資料 CSV", df.to_csv(index=False, encoding="utf-8-sig"), "price_data.csv", "text/csv")

all_trades = []
for name, result in results.items():
    if not result.trades.empty:
        trades = result.trades.copy()
        trades.insert(0, "strategy", name)
        all_trades.append(trades)
trades_download_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
c3.download_button("下載交易紀錄 CSV", trades_download_df.to_csv(index=False, encoding="utf-8-sig"), "trade_records.csv", "text/csv")

equity_df = pd.DataFrame({"time": df["time"]})
for name, result in results.items():
    if not result.equity.empty:
        series = result.equity[["time", "equity"]].rename(columns={"equity": name})
        equity_df = equity_df.merge(series, on="time", how="left")

st.subheader("累積損益曲線")
st.line_chart(equity_df.set_index("time").ffill().fillna(0), height=300)

opt_results = {}
best_params_df = pd.DataFrame()
if run_optimization:
    st.subheader("參數最佳化")
    opt_tabs = st.tabs(selected_strategies)
    for tab, name in zip(opt_tabs, selected_strategies):
        with tab:
            opt_df = optimize_strategy(name, df, base_params)
            opt_results[name] = opt_df
            if opt_df.empty:
                st.info("此策略沒有可顯示的最佳化結果。")
            else:
                st.dataframe(format_metrics(opt_df.head(20)), use_container_width=True)
                st.download_button(
                    f"下載 {name} 最佳化結果 CSV",
                    opt_df.to_csv(index=False, encoding="utf-8-sig"),
                    f"{name}_optimization.csv",
                    "text/csv",
                )
    best_params_df = build_best_parameter_summary(opt_results)
    if not best_params_df.empty:
        st.subheader("各策略最佳參數總表")
        st.dataframe(format_metrics(best_params_df), use_container_width=True)
        st.download_button(
            "下載最佳參數總表 CSV",
            best_params_df.to_csv(index=False, encoding="utf-8-sig"),
            "best_parameter_summary.csv",
            "text/csv",
        )

st.subheader("各策略交易紀錄")
tabs = st.tabs(list(results.keys()))
for tab, name in zip(tabs, results.keys()):
    with tab:
        result = results[name]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("總損益", f"{result.metrics['total_profit']:.2f}")
        c2.metric("交易次數", result.metrics["trade_count"])
        c3.metric("勝率", f"{result.metrics['win_rate']:.2%}")
        c4.metric("最大回撤", f"{result.metrics['max_drawdown']:.2f}")
        c5.metric("報酬回撤比", f"{result.metrics['return_drawdown_ratio']:.2f}")
        if result.trades.empty:
            st.write("此期間沒有完成交易。")
        else:
            st.dataframe(result.trades.tail(100), use_container_width=True)

api_key = get_api_key_from_streamlit(st)

st.subheader("生成式 AI 自動評估")
report = build_ai_report(comparison, api_key=api_key, temperature=temperature, best_params_df=best_params_df)
st.markdown(report)

st.subheader("策略追問聊天室")
if not api_key:
    st.info("目前未設定 DeepSeek API key，因此聊天室不會啟用；已保留上方離線績效評估。")
else:
    st.caption("可直接追問目前回測結果，例如：哪個策略風險較低？MACD 為什麼輸給布林通道？")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.button("清除追問紀錄"):
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("針對目前回測結果提問")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("AI 分析中..."):
                answer = answer_followup_question(
                    comparison,
                    st.session_state.chat_messages,
                    question,
                    api_key=api_key,
                    temperature=temperature,
                    best_params_df=best_params_df,
                )
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

with st.expander("買進持有基準詳細資料"):
    st.json(benchmark.metrics)
