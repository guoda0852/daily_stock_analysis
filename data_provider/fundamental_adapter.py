# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    # Keep pre-tax semantics; skip explicit after-tax plans unless pre-tax marker exists.
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(
    df: pd.DataFrame, stock_code: str, *, require_code: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        if require_code:
            raise ValueError("Market-wide result is missing a stock code column")
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _recent_report_dates(now: Optional[datetime] = None) -> List[str]:
    """Bound report-period fallback to the two most recently completed quarters."""
    now = now or datetime.now()
    quarter_start = datetime(now.year, (now.month - 1) // 3 * 3 + 1, 1)
    dates = []
    for _ in range(2):
        end = quarter_start - timedelta(days=1)
        dates.append(end.strftime("%Y%m%d"))
        quarter_start = datetime(end.year, (end.month - 1) // 3 * 3 + 1, 1)
    return dates


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _earnings_forecast_summary(row: pd.Series) -> str:
    """Read disclosure text by exact column, never an announcement date/rate."""
    for column in ("业绩变动", "业绩变动原因", "预告类型", "预告", "摘要"):
        value = row.get(column)
        if isinstance(value, str) and value.strip().lower() not in ("", "-", "nan", "none", "nat", "<na>"):
            return value.strip()[:200]
    return ""


def _earnings_quick_summary(row: pd.Series) -> str:
    """Summarize AkShare's numeric quick-report columns, excluding metadata."""
    fields = (
        ("营业收入-营业收入", "营业收入", "元"),
        ("营业收入-同比增长", "营收同比", "%"),
        ("净利润-净利润", "净利润", "元"),
        ("净利润-同比增长", "净利润同比", "%"),
        ("每股收益", "每股收益", "元"),
        ("净资产收益率", "净资产收益率", "%"),
    )
    parts = []
    for column, label, unit in fields:
        value = _safe_float(row.get(column))
        if value is not None and math.isfinite(value):
            parts.append(f"{label}{value:.12g}{unit}")
    return "；".join(parts)[:200]


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


def _financial_abstract_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Normalize Sina's metric rows and report-period columns into one row."""
    periods = [c for c in df.columns if re.fullmatch(r"\d{8}", str(c)) and _normalize_report_date(str(c))]
    if not periods:
        return None
    period = max(periods, key=str)
    metrics = df.drop_duplicates(subset=["指标"]).set_index("指标")[period]
    # Include AkShare's unit-bearing labels; exact aliases keep percentages
    # separate from amounts and preserve qualifiers such as weighted ROE.
    aliases = {
        "营业总收入": ["营业总收入", "营业收入", "营业总收入(元)"],
        "归母净利润": ["归母净利润", "归属于母公司股东的净利润", "归属净利润(元)"],
        "经营活动产生的现金流量净额": [
            "经营活动产生的现金流量净额", "经营活动产生的现金流量净额(元)", "经营现金流量净额(元)", "经营现金流量净额",
        ],
        "营业收入同比": ["营业总收入同比增长率", "营业收入同比增长率", "营业总收入同比增长(%)", "营业总收入增长率"],
        "净利润同比": ["净利润同比增长率", "归母净利润同比增长率", "归属净利润同比增长(%)", "归属母公司净利润增长率"],
        "净资产收益率": ["净资产收益率", "净资产收益率(加权)", "净资产收益率(%)", "净资产收益率(加权)(%)", "净资产收益率(ROE)"],
        "毛利率": ["销售毛利率", "毛利率", "销售毛利率(%)"],
    }
    values = {"报告期": str(period)}
    for key, names in aliases.items():
        for name in names:
            value = _safe_float(metrics.get(name))
            if value is not None and math.isfinite(value):
                values[key] = value
                break
    return pd.Series(values)


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
        *,
        stock_code: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty and stock_code is not None:
                    # Market-wide endpoints must contain the requested stock.
                    # A nonempty table for other stocks must not stop fallback.
                    df = _filter_rows_by_code(df, stock_code, require_code=True)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from AkShare with partial tolerance.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # Reuse the provider's market mapping after package initialization;
        # importing it at module scope would cycle through data_provider.base.
        from .akshare_fetcher import _to_sina_tx_symbol

        shareholder_symbol = _to_sina_tx_symbol(stock_code)
        stock_code = _normalize_code(stock_code)
        report_dates = _recent_report_dates()

        # Financial indicators
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            row = _financial_abstract_row(fin_df) if "指标" in fin_df else _extract_latest_row(fin_df, stock_code)
            if row is not None:
                revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                is_wide = "指标" in fin_df
                revenue = _safe_float(
                    row.get("营业总收入") if is_wide else _pick_by_keywords(row, ["营业总收入", "营业收入", "营收"])
                )
                net_profit_parent = _safe_float(
                    row.get("归母净利润") if is_wide else _pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"])
                )
                operating_cash_flow = _safe_float(
                    _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                )
                growth_payload = {
                    "revenue_yoy": revenue_yoy,
                    "net_profit_yoy": profit_yoy,
                    "roe": roe,
                    "gross_margin": gross_margin,
                }
                financial_report_payload = {
                    "report_date": report_date,
                    "revenue": revenue,
                    "net_profit_parent": net_profit_parent,
                    "operating_cash_flow": operating_cash_flow,
                    "roe": roe,
                }
                if any(v is not None for k, v in financial_report_payload.items() if k != "report_date"):
                    result["earnings"]["financial_report"] = financial_report_payload
                if any(v is not None for v in growth_payload.values()):
                    result["growth"] = growth_payload
                if result["growth"] or result["earnings"].get("financial_report"):
                    result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"date": report_date})
            for report_date in report_dates
        ], stock_code=stock_code)
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                summary = _earnings_forecast_summary(row)
                if summary:
                    result["earnings"]["forecast_summary"] = summary
                    result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"date": report_date})
            for report_date in report_dates
        ], stock_code=stock_code)
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                summary = _earnings_quick_summary(row)
                if summary:
                    result["earnings"]["quick_report_summary"] = summary
                    result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Dividend details (cash dividend, pre-tax)
        dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
            ("stock_fhps_detail_em", {"symbol": stock_code}),
            ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
            ("stock_dividend_cninfo", {"symbol": stock_code}),
        ])
        result["errors"].extend(dividend_errors)
        if dividend_df is not None:
            dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {
                "symbol": f"{report_date[:4]}{int(report_date[4:6]) // 3}",
            })
            for report_date in report_dates
        ], stock_code=stock_code)
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": shareholder_symbol, "date": report_date})
            for report_date in report_dates
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        from .akshare_fetcher import _to_sina_tx_symbol

        market = _to_sina_tx_symbol(stock_code)[:2]
        stock_code = _normalize_code(stock_code)
        stock_df, stock_source, stock_errors = self._call_df_candidates([
            ("stock_individual_fund_flow", {"stock": stock_code, "market": market}),
        ])
        result["errors"].extend(stock_errors)
        if stock_df is not None and "日期" in stock_df:
            dates = pd.to_datetime(stock_df["日期"], errors="coerce")
            if dates.notna().any():
                row = stock_df.loc[dates.idxmax()]
                net_inflow = _safe_float(row.get("主力净流入-净额"))
                if net_inflow is not None and math.isfinite(net_inflow):
                    result["stock_flow"] = {
                        "main_net_inflow": net_inflow,
                        "inflow_5d": None,
                        "inflow_10d": None,
                    }
                    result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = "今日主力净流入-净额" if "今日主力净流入-净额" in sector_df else None
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.loc[work_df[flow_col].between(-math.inf, math.inf, inclusive="neither")]
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result


# ----------------------------------------------------------------------
# Hithink（同花顺 Financial-API）基本面适配器
# ----------------------------------------------------------------------
# 覆盖：成长指标（indicators）、财报（income/balance/cash-flow statements）、
# 估值快照（valuations/snapshot）、个股资金流（capital-flow/snapshot）、
# 龙虎榜上榜标记（special-data/dragon-tiger-list）。
# 明确不覆盖（Hithink 无此端点）：业绩预告 / 业绩快报 / 分红明细 /
# 机构持股变化 / 十大流通股东变化 —— 这些块由可选的 Akshare fallback
# 适配器回填（仅补缺失键，Hithink 已给的块不动），source_chain 中注明来源。
# 主力净流入定义：超大单净额 + 大单净额（与东财“主力净流入”口径相近，
# 非同一上游，跨源对比时以 source_chain 为准）。
_HITHINK_BASE_URL = "https://fuyao.aicubes.cn"


def _hithink_thscode(stock_code: str) -> str:
    """本文件内复用：各种代码写法 → Hithink thscode（600519.SH）。"""
    raw = (stock_code or "").strip().upper()
    if not raw:
        return ""
    if raw.startswith(("SH", "SZ", "BJ")):
        return f"{raw[2:]}.{raw[:2]}"
    if "." in raw:
        code, suffix = raw.split(".", 1)
        suffix = suffix.upper()
        if suffix == "SS":
            return f"{code}.SH"
        return f"{code}.{suffix}"
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) != 6:
        return raw
    if digits.startswith(("60", "68", "90")):
        return f"{digits}.SH"
    if digits.startswith(("43", "92")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _report_date_to_hithink(report_date: str) -> Optional[str]:
    """YYYYMMDD → Hithink indicators 报告期格式 yyyy-q（1=一季报…4=年报）。"""
    digits = "".join(ch for ch in str(report_date) if ch.isdigit())
    if len(digits) != 8:
        return None
    year, month = int(digits[0:4]), int(digits[4:6])
    quarter = (month - 1) // 3 + 1
    return f"{year}-{quarter}"


class HithinkFundamentalAdapter:
    """同花顺 Hithink 基本面适配器（fail-open，与 Akshare 版同接口）。"""

    def __init__(self, api_key: str, timeout: int = 30, fallback: Any = None) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ValueError("HithinkFundamentalAdapter 需要 HITHINK_API_KEY")
        self.timeout = timeout
        self.fallback = fallback

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{_HITHINK_BASE_URL}{path}",
                headers={"X-api-key": self.api_key},
                params=params or {},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - fail-open：调用方按 partial 处理
            logger.debug("[HithinkFundamental] 请求失败 %s: %s", path, exc)
            return None
        if not isinstance(payload, dict) or payload.get("code") != 0:
            logger.debug(
                "[HithinkFundamental] 业务错误 %s: code=%s",
                path,
                payload.get("code") if isinstance(payload, dict) else "?",
            )
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def _latest_indicators(self, thscode: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """取最近两期报告期的 indicators，返回（指标字典，报告期）。"""
        for report_date in _recent_report_dates():
            report = _report_date_to_hithink(report_date)
            if not report:
                continue
            data = self._get(
                "/api/a-share/financials/indicators",
                {"thscode": thscode, "report": report},
            )
            if not data:
                continue
            flat: Dict[str, Any] = {}
            for block in data.get("abilities") or []:
                for indicator in block.get("indicators") or []:
                    index_id = indicator.get("index_id")
                    if index_id:
                        flat[index_id] = _safe_float(indicator.get("value"))
            if flat:
                return flat, report
        return None, None

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }
        thscode = _hithink_thscode(stock_code)
        if not thscode:
            result["errors"].append(f"无法解析代码 {stock_code}")
            return result

        indicators, report = self._latest_indicators(thscode)
        if indicators:
            growth_payload = {
                "revenue_yoy": indicators.get("operating_income_yoy_growth_ratio"),
                "net_profit_yoy": indicators.get("net_profit_yoy_growth_ratio"),
                "roe": indicators.get("index_weighted_avg_roe"),
                "gross_margin": indicators.get("sale_gross_margin"),
            }
            if any(v is not None for v in growth_payload.values()):
                result["growth"] = growth_payload
                result["source_chain"].append(f"growth:hithink_indicators@{report}")

        income = self._get(
            "/api/a-share/financials/income-statements",
            {"thscode": thscode, "period": "quarterly", "limit": 1},
        )
        cashflow = self._get(
            "/api/a-share/financials/cash-flow-statements",
            {"thscode": thscode, "period": "quarterly", "limit": 1},
        )
        income_item = (income.get("item") or [None])[0] if income else None
        cashflow_item = (cashflow.get("item") or [None])[0] if cashflow else None
        if income_item or cashflow_item:
            report_date = None
            if income_item:
                report_ms = income_item.get("report_date_ms") or income_item.get("period_end_ms")
                if report_ms:
                    try:
                        report_date = datetime.fromtimestamp(
                            int(report_ms) / 1000
                        ).strftime("%Y%m%d")
                    except (TypeError, ValueError):
                        report_date = None
            roe = (indicators or {}).get("index_weighted_avg_roe")
            financial_report_payload = {
                "report_date": report_date,
                "revenue": _safe_float((income_item or {}).get("operating_income")),
                "net_profit_parent": _safe_float(
                    (income_item or {}).get("parent_holder_net_profit")
                ),
                "operating_cash_flow": _safe_float(
                    (cashflow_item or {}).get("act_cash_flow_net")
                ),
                "roe": roe,
            }
            if any(
                v is not None for k, v in financial_report_payload.items() if k != "report_date"
            ):
                result["earnings"]["financial_report"] = financial_report_payload
                result["source_chain"].append("earnings_report:hithink_financials")

        valuation = self._get(
            "/api/a-share/valuations/snapshot", {"thscodes": thscode}
        )
        vitems = (valuation.get("item") or []) if valuation else []
        if vitems:
            item = vitems[0]
            valuation_payload = {
                "pe_ttm": _safe_float(item.get("pe_ttm")),
                "pe_mrq": _safe_float(item.get("pe_mrq")),
                "pb_mrq": _safe_float(item.get("pb_mrq")),
                "ps_ttm": _safe_float(item.get("ps_ttm")),
                "pcf_ttm": _safe_float(item.get("pcf_ttm")),
            }
            if any(v is not None for v in valuation_payload.values()):
                result["earnings"]["valuation"] = valuation_payload
                result["source_chain"].append("valuation:hithink_valuations")

        # 回填 Hithink 没有的块（业绩预告 / 快报 / 分红 / 机构持股），仅补缺失键。
        if self.fallback is not None:
            try:
                fallback_bundle = self.fallback.get_fundamental_bundle(stock_code)
            except Exception as exc:  # noqa: BLE001 - fail-open
                logger.debug("[HithinkFundamental] fallback bundle 失败: %s", exc)
                fallback_bundle = {}
            if isinstance(fallback_bundle, dict):
                for key in ("forecast_summary", "quick_report_summary", "dividend"):
                    if key not in result["earnings"] and fallback_bundle.get("earnings", {}).get(key):
                        result["earnings"][key] = fallback_bundle["earnings"][key]
                if not result["growth"] and fallback_bundle.get("growth"):
                    result["growth"] = fallback_bundle["growth"]
                if not result["institution"] and fallback_bundle.get("institution"):
                    result["institution"] = fallback_bundle["institution"]
                for entry in fallback_bundle.get("source_chain", []):
                    if entry not in result["source_chain"]:
                        result["source_chain"].append(entry)
                result["errors"].extend(fallback_bundle.get("errors", []))

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }
        thscode = _hithink_thscode(stock_code)
        if not thscode:
            return result
        data = self._get("/api/a-share/capital-flow/snapshot", {"thscode": thscode})
        if data:
            def _net(bucket: Any) -> Optional[float]:
                if not isinstance(bucket, dict):
                    return None
                return _safe_float(bucket.get("net_amount"))

            super_large = _net(data.get("super_large"))
            large = _net(data.get("large"))
            medium = _net(data.get("medium"))
            small = _net(data.get("small"))
            main_net = None
            if super_large is not None or large is not None:
                main_net = (super_large or 0.0) + (large or 0.0)
            if main_net is not None:
                result["stock_flow"] = {
                    "main_net_inflow": main_net,
                    "super_large_net": super_large,
                    "large_net": large,
                    "medium_net": medium,
                    "small_net": small,
                }
                result["source_chain"].append("capital_stock:hithink_capital_flow")
        if self.fallback is not None:
            try:
                fallback_flow = self.fallback.get_capital_flow(stock_code, top_n=top_n)
            except Exception as exc:  # noqa: BLE001 - fail-open
                logger.debug("[HithinkFundamental] fallback capital_flow 失败: %s", exc)
                fallback_flow = {}
            if isinstance(fallback_flow, dict):
                if not result["stock_flow"] and fallback_flow.get("stock_flow"):
                    result["stock_flow"] = fallback_flow["stock_flow"]
                sector = fallback_flow.get("sector_rankings") or {}
                if (sector.get("top") or sector.get("bottom")) and not (
                    result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"]
                ):
                    result["sector_rankings"] = sector
                for entry in fallback_flow.get("source_chain", []):
                    if entry not in result["source_chain"]:
                        result["source_chain"].append(entry)
                result["errors"].extend(fallback_flow.get("errors", []))
        has_content = bool(
            result["stock_flow"]
            or result["sector_rankings"]["top"]
            or result["sector_rankings"]["bottom"]
        )
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }
        target = _normalize_code(stock_code)
        if not target:
            return result
        # 取近一年交易日，筛出 lookback 窗口内最近 5 个交易日逐日查榜（调用量上限 5 次）。
        trading_days: List[str] = []
        calendar = self._get("/api/a-share/calendar/trading-days")
        if calendar:
            for entry in calendar.get("item") or []:
                date_str = str(entry.get("date") or "")
                if len(date_str) == 8:
                    trading_days.append(date_str)
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_days = [
            d
            for d in trading_days
            if start <= datetime.strptime(d, "%Y%m%d") <= now
        ][-5:]
        if not recent_days:
            recent_days = [None]  # 日历取不到时只查服务端最近可用交易日
        matched_dates: List[str] = []
        for day in recent_days:
            params: Dict[str, Any] = {"board_type": "all"}
            if day:
                params["date"] = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"
            data = self._get("/api/a-share/special-data/dragon-tiger-list", params)
            if not data:
                continue
            result["source_chain"].append("dragon_tiger:hithink_dragon_tiger_list")
            codes = {
                _normalize_code(item.get("thscode") or item.get("ticker"))
                for item in data.get("stock_items") or []
            }
            if target in codes and day:
                matched_dates.append(day)
            elif target in codes and not day:
                matched_dates.append(now.strftime("%Y%m%d"))
        if matched_dates:
            result["is_on_list"] = True
            result["recent_count"] = len(matched_dates)
            result["latest_date"] = max(matched_dates)
            result["status"] = "ok"
        elif result["source_chain"]:
            result["status"] = "ok"
        return result
