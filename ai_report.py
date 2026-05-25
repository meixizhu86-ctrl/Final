import json
import os

import pandas as pd
import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def build_rule_report(metrics_df: pd.DataFrame, best_params_df: pd.DataFrame | None = None) -> str:
    if metrics_df.empty:
        return "目前沒有可評估的交易結果。請先選擇至少一個策略。"

    ranked = metrics_df.sort_values(["score", "total_profit"], ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    stable = ranked.sort_values(["max_drawdown", "total_profit"], ascending=[True, False]).iloc[0]

    lines = [
        "### 離線績效評估",
        f"- 綜合分數最高的是 **{best['strategy']}**，總損益為 `{best['total_profit']:.2f}`，勝率為 `{best['win_rate']:.2%}`，最大回撤為 `{best['max_drawdown']:.2f}`。",
        f"- 風險相對較低的是 **{stable['strategy']}**，最大回撤為 `{stable['max_drawdown']:.2f}`，報酬回撤比為 `{stable['return_drawdown_ratio']:.2f}`。",
        f"- 表現較弱的是 **{worst['strategy']}**，總損益為 `{worst['total_profit']:.2f}`。",
        "- 評估分數同時考慮總損益、勝率、最大回撤與報酬回撤比，避免只追求高報酬而忽略風險。",
    ]
    if best_params_df is not None and not best_params_df.empty:
        opt_best = best_params_df.iloc[0]
        lines.append(f"- 參數最佳化後，目前最佳組合來自 **{opt_best['strategy']}**，最佳參數為 `{opt_best['best_params']}`。")
    lines.append("- 這是離線規則式分析；設定 DeepSeek API key 後會改由生成式 AI 產生更完整的策略比較。")
    return "\n\n".join(lines)


def _post_deepseek(messages: list[dict], api_key: str, temperature: float) -> str:
    payload = {"model": DEEPSEEK_MODEL, "temperature": temperature, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(DEEPSEEK_BASE_URL, headers=headers, data=json.dumps(payload), timeout=40)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_ai_report(
    metrics_df: pd.DataFrame,
    api_key: str | None,
    temperature: float = 0.3,
    best_params_df: pd.DataFrame | None = None,
) -> str:
    if metrics_df.empty:
        return "目前沒有可評估的交易結果。"
    if not api_key:
        return build_rule_report(metrics_df, best_params_df)

    best_params_text = ""
    if best_params_df is not None and not best_params_df.empty:
        best_params_text = "\n\n參數最佳化摘要 JSON：\n" + best_params_df.to_json(orient="records", force_ascii=False)

    messages = [
        {
            "role": "system",
            "content": "你是台股量化交易課程助教。請用繁體中文客觀比較策略績效，避免承諾未來獲利。",
        },
        {
            "role": "user",
            "content": (
                "請根據以下策略回測績效表與最佳化摘要，分析各策略的報酬、風險、勝率、最大回撤、報酬回撤比與改善方向。"
                "請輸出 5 到 7 點條列式重點，指出最推薦策略、風險較低策略，以及可能過度最佳化的風險。"
                "最後補一句歷史回測不代表未來績效的提醒。\n\n"
                "策略績效表 JSON：\n"
                + metrics_df.to_json(orient="records", force_ascii=False)
                + best_params_text
            ),
        },
    ]

    try:
        return _post_deepseek(messages, api_key, temperature)
    except Exception as exc:
        return build_rule_report(metrics_df, best_params_df) + f"\n\nDeepSeek API 暫時無法使用，已改用離線分析。錯誤摘要：`{exc}`"


def answer_followup_question(
    metrics_df: pd.DataFrame,
    chat_history: list[dict],
    question: str,
    api_key: str | None,
    temperature: float = 0.3,
    best_params_df: pd.DataFrame | None = None,
) -> str:
    if not api_key:
        return "目前尚未設定 DeepSeek API key，所以無法即時對話。請在 Streamlit Cloud Secrets 設定 DEEPSEEK_API_KEY。"

    system_prompt = (
        "你是台股量化交易課程助教。使用者正在查看 Streamlit 回測 APP。"
        "請根據提供的績效表和最佳化摘要回答追問，使用繁體中文，回答要具體、簡潔、可行。"
        "不可承諾未來獲利，必要時提醒這只是歷史回測。"
    )
    context = "目前回測績效表 JSON：\n" + metrics_df.to_json(orient="records", force_ascii=False)
    if best_params_df is not None and not best_params_df.empty:
        context += "\n\n目前最佳化摘要 JSON：\n" + best_params_df.to_json(orient="records", force_ascii=False)
    context += "\n\n請回答使用者的追問。"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": question})

    try:
        return _post_deepseek(messages, api_key, temperature)
    except Exception as exc:
        return f"DeepSeek API 暫時無法回答，請稍後再試。錯誤摘要：`{exc}`"


def get_api_key_from_streamlit(st_module=None) -> str | None:
    if st_module is not None:
        try:
            key = st_module.secrets.get("DEEPSEEK_API_KEY")
            if key:
                return key
        except Exception:
            pass
    return os.getenv("DEEPSEEK_API_KEY")
