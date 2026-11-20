"""同花顺 Hithink Financial-API 数据源适配器。

数据来源：同花顺官方 Financial-API（https://fuyao.aicubes.cn），
REST + ``X-api-key`` 鉴权，Key 由环境变量 ``HITHINK_API_KEY`` 注入
（仓库 Secrets，见 .github/workflows 补丁说明）。

覆盖能力（与本仓库取数需求的映射见 docs/CHANGELOG 与交付说明）：
- 个股日 K：``GET /api/a-share/prices/historical``（1d，adjust 可配，
  默认 forward 前复权；TickFlow 默认 none，链内各源口径本就不统一）
- 实时行情：``GET /api/a-share/prices/snapshot``（批量 thscode）
- 指数行情 / 日 K：``GET /api/a-share-index/prices/snapshot|historical``
- 大盘涨跌家数 / 成交额：全市场快照分页汇总（limit/offset）
- 涨停池 / 跌停池 / 炸板池：``GET /api/a-share/special-data/limit-up-pool``
  / ``limit-down-pool`` / ``limit-break-pool``，涨停强度：
  ``GET /api/a-share/special-data/limit-up-statistics``
- 板块 / 概念涨跌榜：同花顺行业指数 / 概念指数目录
  （``GET /api/a-share-index/catalog/ths-index-list``）+ 指数快照排序
- 人气股：``GET /api/a-share/special-data/hot-stock-list``
- 估值快照：``GET /api/a-share/valuations/snapshot``（PE TTM/MRQ、PB、PS、PCF）
- 股票名称 / 代码表：``GET /api/a-share/valuations/snapshot``（含 name）、
  ``GET /api/meta/tickers/list``

明确不覆盖（Hithink 不提供，保留现有 fallback 链）：
- 分钟 K / tick / Level-2、港美股行情、宏观数据、公告与研报原文、
  业绩预告 / 业绩快报 / 分红明细 / 机构持股 / 十大流通股东
  （基本面缺失块由 HithinkFundamentalAdapter 回填 Akshare，见
  data_provider/fundamental_adapter.py）。
- 筹码分布：Hithink 只有筹码形态筛选（chip-shape），不提供获利盘比例 /
  平均成本等分布指标；ENABLE_CHIP_DISTRIBUTION 仍走原有链路。

注意：沙箱 DNS 无法解析 fuyao.aicubes.cn，本地无法实测；
真实调用只能走 GitHub Actions（runner 出网正常）。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, normalize_stock_code
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote

logger = logging.getLogger(__name__)

BASE_URL = "https://fuyao.aicubes.cn"
_SHANGHAI = timezone(timedelta(hours=8))

# 与 TickFlowFetcher 保持一致的主流指数表（thscode, 6 位代码, 名称）。
_CN_MAIN_INDEX_QUOTES = (
    ("000001.SH", "000001", "上证指数"),
    ("399001.SZ", "399001", "深证成指"),
    ("399006.SZ", "399006", "创业板指"),
    ("000688.SH", "000688", "科创50"),
    ("000016.SH", "000016", "上证50"),
    ("000300.SH", "000300", "沪深300"),
)

_HITHINK_KLINE_ADJUST_VALUES = {"forward", "backward", "none"}
_SNAPSHOT_PAGE_SIZE = 100
_MAX_SNAPSHOT_PAGES = 120  # 全市场分页安全上限（约 5400 只 ≈ 55 页）
_INDEX_SNAPSHOT_CHUNK = 50  # 指数快照批量 thscode 分组大小


def _to_thscode(stock_code: str) -> str:
    """把本仓库各种代码写法统一成 Hithink thscode（600519.SH / 000001.SZ）。"""
    raw = (stock_code or "").strip().upper()
    if not raw:
        return ""
    if raw.startswith("SH") or raw.startswith("SZ") or raw.startswith("BJ"):
        market = raw[:2]
        digits = raw[2:]
        return f"{digits}.{market}"
    if "." in raw:
        code, suffix = raw.split(".", 1)
        suffix = suffix.upper()
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{code}.{suffix}"
        # yfinance 风格 600519.SS / 000001.SZ
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


def _date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    """YYYY-MM-DD → Asia/Shanghai 当日 00:00（或 23:59:59.999）的毫秒戳。"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_SHANGHAI)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: Any) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=_SHANGHAI).strftime("%Y-%m-%d")


class HithinkFetcher(BaseFetcher):
    """同花顺 Hithink Financial-API 适配器（A 股）。"""

    name = "HithinkFetcher"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        kline_adjust: str = "forward",
        batch_daily_enabled: bool = True,
        batch_size: int = 10,
        priority: int = 0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ValueError("HithinkFetcher 需要 HITHINK_API_KEY")
        self.timeout = timeout
        self.kline_adjust = (
            kline_adjust.strip().lower()
            if isinstance(kline_adjust, str) and kline_adjust.strip().lower() in _HITHINK_KLINE_ADJUST_VALUES
            else "forward"
        )
        self.batch_daily_enabled = batch_daily_enabled
        self.batch_size = max(1, int(batch_size or 10))
        self.priority = priority

    # ------------------------------------------------------------------
    # 底层请求
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{BASE_URL}{path}"
        try:
            resp = requests.get(
                url,
                headers={"X-api-key": self.api_key},
                params=params or {},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(f"[HithinkFetcher] 请求失败 {path}: {exc}") from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataFetchError(f"[HithinkFetcher] 非 JSON 响应 {path}") from exc
        if not isinstance(payload, dict) or payload.get("code") != 0:
            message = payload.get("message") if isinstance(payload, dict) else None
            raise DataFetchError(
                f"[HithinkFetcher] 业务错误 {path}: code={payload.get('code') if isinstance(payload, dict) else '?'} message={message}"
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise DataFetchError(f"[HithinkFetcher] 响应无 data 字段 {path}")
        return data

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # BaseFetcher 抽象方法：日 K
    # ------------------------------------------------------------------
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        thscode = _to_thscode(stock_code)
        if not thscode:
            raise DataFetchError(f"[HithinkFetcher] 无法解析代码 {stock_code}")
        data = self._get(
            "/api/a-share/prices/historical",
            {
                "thscode": thscode,
                "interval": "1d",
                "start": _date_to_ms(start_date),
                "end": _date_to_ms(end_date, end_of_day=True),
                "adjust": self.kline_adjust,
            },
        )
        items = data.get("item") or []
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items)

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", *STANDARD_COLUMNS])
        raw = df.copy()
        normalized = pd.DataFrame()
        if "date_ms" in raw.columns:
            normalized["date"] = pd.to_datetime(
                pd.to_numeric(raw["date_ms"], errors="coerce"), unit="ms", utc=True
            ).dt.tz_convert(_SHANGHAI).dt.strftime("%Y-%m-%d")
        elif "date" in raw.columns:
            normalized["date"] = raw["date"]
        else:
            return pd.DataFrame(columns=["code", *STANDARD_COLUMNS])
        normalized["code"] = normalize_stock_code(stock_code)
        for column, source in (
            ("open", "open_price"),
            ("high", "high_price"),
            ("low", "low_price"),
            ("close", "close_price"),
            ("volume", "volume"),
            ("amount", "turnover"),
        ):
            normalized[column] = pd.to_numeric(raw.get(source), errors="coerce")
        # Hithink volume 已是股数（与标准口径一致，无需像 TickFlow 那样手→股换算）。
        # 历史 K 线不返回涨跌幅，由收盘价序列推导（与 TickFlow _normalize_data 同款兜底）。
        close = pd.to_numeric(normalized["close"], errors="coerce")
        normalized["pct_chg"] = close.pct_change().fillna(0.0) * 100.0
        normalized = normalized.dropna(subset=["date", "close", "volume"])
        normalized = normalized.sort_values("date", ascending=True).reset_index(drop=True)
        return normalized[["code", *STANDARD_COLUMNS]]

    # ------------------------------------------------------------------
    # 实时行情 / 名称 / 代码表
    # ------------------------------------------------------------------
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        thscode = _to_thscode(stock_code)
        if not thscode:
            return None
        try:
            data = self._get("/api/a-share/prices/snapshot", {"thscodes": thscode})
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 实时行情失败 %s: %s", stock_code, exc)
            return None
        items = data.get("item") or []
        if not items:
            return None
        item = items[0]
        price = self._safe_float(item.get("last_price"))
        if price is None:
            return None
        name, pe_ttm, pb_mrq = "", None, None
        try:
            valuation = self._get("/api/a-share/valuations/snapshot", {"thscodes": thscode})
            vitems = valuation.get("item") or []
            if vitems:
                name = str(vitems[0].get("name") or "")
                pe_ttm = self._safe_float(vitems[0].get("pe_ttm"))
                pb_mrq = self._safe_float(vitems[0].get("pb_mrq"))
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 估值快照失败 %s: %s", stock_code, exc)
        volume = self._safe_float(item.get("volume"))
        return UnifiedRealtimeQuote(
            code=normalize_stock_code(stock_code),
            name=name,
            source=RealtimeSource.HITHINK,
            price=price,
            change_pct=self._safe_float(item.get("price_change_ratio_pct")),
            change_amount=self._safe_float(item.get("price_change")),
            volume=int(volume) if volume is not None else None,
            amount=self._safe_float(item.get("turnover")),
            open_price=self._safe_float(item.get("open_price")),
            high=self._safe_float(item.get("high_price")),
            low=self._safe_float(item.get("low_price")),
            pre_close=self._safe_float(item.get("prev_price")),
            pe_ratio=pe_ttm,
            pb_ratio=pb_mrq,
            market="cn",
            currency="CNY",
        )

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        thscode = _to_thscode(stock_code)
        if not thscode:
            return None
        try:
            data = self._get("/api/a-share/valuations/snapshot", {"thscodes": thscode})
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 名称查询失败 %s: %s", stock_code, exc)
            return None
        items = data.get("item") or []
        if not items:
            return None
        name = str(items[0].get("name") or "").strip()
        return name or None

    def get_stock_list(self, market: Optional[str] = None) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["code", "name", "industry", "area", "market"])
        rows: List[Dict[str, Any]] = []
        offset = 0
        limit = 1000
        try:
            while True:
                data = self._get(
                    "/api/meta/tickers/list",
                    {"asset_type": "a-share", "limit": limit, "offset": offset},
                )
                items = data.get("item") or []
                for entry in items:
                    thscode = str(entry.get("thscode") or "")
                    if not thscode:
                        continue
                    code, _, suffix = thscode.partition(".")
                    rows.append(
                        {
                            "code": normalize_stock_code(code),
                            "name": str(entry.get("name") or ""),
                            "industry": "",
                            "area": "",
                            "market": suffix,
                        }
                    )
                if len(items) < limit:
                    break
                offset += limit
                if offset > 20000:  # 安全上限
                    break
        except DataFetchError as exc:
            logger.warning("[HithinkFetcher] 代码表获取失败: %s", exc)
            return empty
        if not rows:
            return empty
        return pd.DataFrame(rows, columns=["code", "name", "industry", "area", "market"])

    # ------------------------------------------------------------------
    # 指数 / 大盘
    # ------------------------------------------------------------------
    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        if region != "cn":
            return None
        thscodes = ",".join(symbol for symbol, _, _ in _CN_MAIN_INDEX_QUOTES)
        try:
            data = self._get("/api/a-share-index/prices/snapshot", {"thscodes": thscodes})
        except DataFetchError as exc:
            logger.warning("[HithinkFetcher] 指数行情失败: %s", exc)
            return None
        items = data.get("item") or []
        if not items:
            return None
        by_thscode = {str(item.get("thscode", "")).upper(): item for item in items if item}
        results: List[Dict[str, Any]] = []
        for symbol, code, name in _CN_MAIN_INDEX_QUOTES:
            item = by_thscode.get(symbol)
            if not item:
                continue
            current = self._safe_float(item.get("last_price")) or 0.0
            prev_close = self._safe_float(item.get("prev_price")) or 0.0
            change = self._safe_float(item.get("price_change"))
            if change is None:
                change = current - prev_close if current or prev_close else 0.0
            high = self._safe_float(item.get("high_price")) or 0.0
            low = self._safe_float(item.get("low_price")) or 0.0
            amplitude = (high - low) / prev_close * 100 if prev_close > 0 else 0.0
            results.append(
                {
                    "code": code,
                    "name": name,
                    "current": current,
                    "change": change,
                    "change_pct": self._safe_float(item.get("price_change_ratio_pct")) or 0.0,
                    "open": self._safe_float(item.get("open_price")) or 0.0,
                    "high": high,
                    "low": low,
                    "prev_close": prev_close,
                    "volume": self._safe_float(item.get("volume")) or 0.0,
                    "amount": self._safe_float(item.get("turnover")) or 0.0,
                    "amplitude": amplitude,
                }
            )
        return results or None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """全市场快照分页汇总：上涨/下跌/平盘家数与两市成交额。

        涨停 / 跌停家数优先用涨停强度统计与跌停池总数；取不到时用
        ±9.9% 涨跌幅阈值近似（ST 股 5% 涨跌停会被低估，日志中注明）。
        """
        up_count = down_count = flat_count = 0
        total_amount = 0.0
        offset = 0
        pages = 0
        try:
            while pages < _MAX_SNAPSHOT_PAGES:
                data = self._get(
                    "/api/a-share/prices/snapshot",
                    {"limit": _SNAPSHOT_PAGE_SIZE, "offset": offset},
                )
                items = data.get("item") or []
                for item in items:
                    pct = self._safe_float(item.get("price_change_ratio_pct"))
                    turnover = self._safe_float(item.get("turnover")) or 0.0
                    total_amount += turnover
                    if pct is None:
                        flat_count += 1
                    elif pct > 0:
                        up_count += 1
                    elif pct < 0:
                        down_count += 1
                    else:
                        flat_count += 1
                pages += 1
                if len(items) < _SNAPSHOT_PAGE_SIZE:
                    break
                offset += _SNAPSHOT_PAGE_SIZE
        except DataFetchError as exc:
            logger.warning("[HithinkFetcher] 大盘统计快照分页失败: %s", exc)
            return None
        if up_count + down_count + flat_count == 0:
            return None

        limit_up_count: Optional[int] = None
        limit_down_count: Optional[int] = None
        try:
            stats = self._get("/api/a-share/special-data/limit-up-statistics")
            limit_up_count = stats.get("limit_up_stock_count")
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 涨停统计失败，改用阈值近似: %s", exc)
        try:
            down_pool = self._get(
                "/api/a-share/special-data/limit-down-pool",
                {"page": 1, "size": 1},
            )
            pagination = down_pool.get("pagination") or {}
            limit_down_count = pagination.get("total")
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 跌停池失败，改用阈值近似: %s", exc)
        if limit_up_count is None or limit_down_count is None:
            # 阈值近似：需要第二遍分页，成本高，仅在池接口失败时走。
            logger.warning("[HithinkFetcher] 涨跌停家数改用 ±9.9% 阈值近似（ST 股 5% 会被低估）")
            approx_up = approx_down = 0
            offset, pages = 0, 0
            try:
                while pages < _MAX_SNAPSHOT_PAGES:
                    data = self._get(
                        "/api/a-share/prices/snapshot",
                        {"limit": _SNAPSHOT_PAGE_SIZE, "offset": offset},
                    )
                    items = data.get("item") or []
                    for item in items:
                        pct = self._safe_float(item.get("price_change_ratio_pct"))
                        if pct is None:
                            continue
                        if pct >= 9.9:
                            approx_up += 1
                        elif pct <= -9.9:
                            approx_down += 1
                    pages += 1
                    if len(items) < _SNAPSHOT_PAGE_SIZE:
                        break
                    offset += _SNAPSHOT_PAGE_SIZE
            except DataFetchError as exc:
                logger.warning("[HithinkFetcher] 阈值近似分页失败: %s", exc)
            if limit_up_count is None:
                limit_up_count = approx_up
            if limit_down_count is None:
                limit_down_count = approx_down
        return {
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "limit_up_count": int(limit_up_count or 0),
            "limit_down_count": int(limit_down_count or 0),
            "total_amount": total_amount,
        }

    # ------------------------------------------------------------------
    # 板块 / 概念涨跌榜（同花顺行业指数 / 概念指数快照排序）
    # ------------------------------------------------------------------
    def _get_index_rankings(
        self, tag: str, n: int, source_label: str
    ) -> Optional[Tuple[List[Dict], List[Dict]]]:
        try:
            catalog = self._get(
                "/api/a-share-index/catalog/ths-index-list", {"tag": tag}
            )
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 指数目录失败 tag=%s: %s", tag, exc)
            return None
        items = catalog.get("item") or []
        if not items:
            return None
        thscodes = [str(item.get("thscode", "")).strip() for item in items]
        thscodes = [code for code in thscodes if code]
        names = {
            str(item.get("thscode", "")).strip(): str(item.get("name") or "")
            for item in items
        }
        ranked: List[Dict[str, Any]] = []
        for start in range(0, len(thscodes), _INDEX_SNAPSHOT_CHUNK):
            chunk = thscodes[start : start + _INDEX_SNAPSHOT_CHUNK]
            try:
                data = self._get(
                    "/api/a-share-index/prices/snapshot",
                    {"thscodes": ",".join(chunk)},
                )
            except DataFetchError as exc:
                logger.debug("[HithinkFetcher] 指数快照失败 tag=%s: %s", tag, exc)
                continue
            for item in data.get("item") or []:
                pct = self._safe_float(item.get("price_change_ratio_pct"))
                if pct is None:
                    continue
                thscode = str(item.get("thscode", "")).strip()
                ranked.append(
                    {
                        "name": names.get(thscode, thscode),
                        "change_pct": round(pct, 4),
                        "source": source_label,
                        "constituent_count": None,
                    }
                )
        if not ranked:
            return None
        ranked.sort(key=lambda row: row["change_pct"], reverse=True)
        return ranked[:n], list(reversed(ranked[-n:]))

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """行业板块涨跌榜：同花顺行业指数（tag=industry）快照排序。"""
        return self._get_index_rankings("industry", n, "hithink_industry")

    def get_concept_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """概念涨跌榜：同花顺概念指数（tag=cn_concept）快照排序。"""
        return self._get_index_rankings("cn_concept", n, "hithink_concept")

    # ------------------------------------------------------------------
    # 人气股 / 涨停池
    # ------------------------------------------------------------------
    def get_hot_stocks(self, n: int = 10) -> Optional[List[Dict[str, Any]]]:
        try:
            data = self._get(
                "/api/a-share/special-data/hot-stock-list",
                {"period": "day", "limit": max(1, min(int(n), 100))},
            )
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 人气股失败: %s", exc)
            return None
        rows: List[Dict[str, Any]] = []
        for item in data.get("item") or []:
            ticker = str(item.get("ticker") or "").strip()
            if not ticker:
                continue
            rows.append(
                {
                    "code": normalize_stock_code(ticker),
                    "name": str(item.get("name") or ""),
                    "rank": item.get("rank"),
                    "heat": item.get("heat"),
                    "rank_change": item.get("rank_change"),
                    "rank_trend": item.get("rank_trend"),
                }
            )
        return rows or None

    def get_limit_up_pool(
        self,
        date: Optional[str] = None,
        n: int = 20,
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, Any] = {
            "page": 1,
            "size": max(1, min(int(n), 200)),
            "sort_field": "limit_up_time",
            "sort_dir": "desc",
        }
        if date:
            digits = "".join(ch for ch in str(date) if ch.isdigit())
            if len(digits) == 8:
                try:
                    params["date_ms"] = _date_to_ms(
                        f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
                    )
                except ValueError:
                    pass
        try:
            data = self._get("/api/a-share/special-data/limit-up-pool", params)
        except DataFetchError as exc:
            logger.debug("[HithinkFetcher] 涨停池失败: %s", exc)
            return None
        rows: List[Dict[str, Any]] = []
        for item in data.get("item") or []:
            ticker = str(item.get("ticker") or "").strip()
            if not ticker:
                continue
            rows.append(
                {
                    "code": normalize_stock_code(ticker),
                    "name": str(item.get("name") or ""),
                    "change_pct": self._safe_float(item.get("price_change_ratio_pct")),
                    "price": self._safe_float(item.get("last_price")),
                    "amount": self._safe_float(item.get("turnover")),
                    "turnover_rate": self._safe_float(item.get("turnover_ratio_pct")),
                    "seal_amount": self._safe_float(item.get("seal_money")),
                    "first_limit_time": str(item.get("first_limit_time") or ""),
                    "limit_up_reason": item.get("limit_up_reason"),
                    "continue_days": item.get("continue_day_cnt"),
                }
            )
        return rows or None
