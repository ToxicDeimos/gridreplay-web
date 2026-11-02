"""Análisis de un export de Pionex sin servidor (pensado para Pyodide).

El analizador de la web ejecuta este módulo en el navegador del usuario: el
historial nunca sale de su equipo. Las descargas van por una función `get(url)`
asíncrona que devuelve (estado HTTP, cuerpo); en el navegador es `fetch`, en
los tests puede ser cualquier otra cosa. La simulación y la comparación son las
mismas que en `gridreplay pionex` (`compare_bot`), así que los resultados
coinciden con la validación publicada.

Mismas fuentes que `data.py`: archivo mensual de Binance verificado por
checksum, ficheros diarios para el mes en curso y la API para los últimos días.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlencode

import pandas as pd

from gridreplay.data import (
    BASE_URLS,
    COLUMNS,
    FUNDING_API,
    KLINE_APIS,
    DataNotAvailable,
    _months_between,
    _to_utc,
    _utc_range,
    parse_funding_csv,
    parse_kline_csv,
)
from gridreplay.fees import fees_for
from gridreplay.pionex import PionexBot, detect_bots, load_export
from gridreplay.robust import Template, run_windows, summarize
from gridreplay.validate import compare_bot, sim_window

Get = Callable[[str], Awaitable[tuple[int, bytes]]]
Progress = Callable[[dict], None]


class Loader:
    """Descarga velas y funding de futuros con caché en memoria por mes."""

    def __init__(self, get: Get, now: datetime | None = None) -> None:
        self.get = get
        self.now = now or datetime.now(UTC)
        self._months: dict[tuple, asyncio.Task] = {}

    async def _zip(self, url: str) -> bytes:
        status, body = await self.get(url)
        if status == 404:
            raise DataNotAvailable(url)
        if status != 200:
            raise OSError(f"HTTP {status} en {url}")
        status, check = await self.get(url + ".CHECKSUM")
        if status != 200:
            raise OSError(f"HTTP {status} en {url}.CHECKSUM")
        if hashlib.sha256(body).hexdigest() != check.decode().split()[0].strip().lower():
            raise OSError(f"Checksum incorrecto para {url}")
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            return zf.read(zf.namelist()[0])

    async def _json(self, base: str, params: dict) -> list:
        url = f"{base}?{urlencode(params)}"
        status, body = await self.get(url)
        if status != 200:
            raise OSError(f"HTTP {status} en {url}")
        return json.loads(body)

    def _once(self, key: tuple, make) -> asyncio.Task:
        # Varios bots del mismo par comparten meses: se descargan una sola vez.
        if key not in self._months:
            self._months[key] = asyncio.ensure_future(make())
        return self._months[key]

    async def _kline_month(self, symbol: str, year: int, month: int) -> list[pd.DataFrame]:
        base = f"{BASE_URLS['futures']}/{{}}/klines/{symbol}/1m/{symbol}-1m-{{}}.zip"
        tag = f"{year:04d}-{month:02d}"
        try:
            return [parse_kline_csv(await self._zip(base.format("monthly", tag)))]
        except DataNotAvailable:
            pass
        frames = []
        day = date(year, month, 1)
        while day.month == month and day < self.now.date():
            try:
                frames.append(parse_kline_csv(await self._zip(base.format("daily", day))))
            except DataNotAvailable:
                break
            day += timedelta(days=1)
        return frames

    async def _recent_klines(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows: list[list] = []
        cursor, stop = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        while cursor < stop:
            batch = await self._json(
                KLINE_APIS["futures"],
                {
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": cursor,
                    "endTime": stop - 1,
                    "limit": 1500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + 60_000
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS)
        out = pd.DataFrame({"ts": _to_utc(df["ts"])})
        for col in COLUMNS[1:]:
            out[col] = df[col].astype("float64").to_numpy()
        return out[out["ts"] + pd.Timedelta(minutes=1) <= self.now]

    async def candles(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        start, end, last_day = _utc_range(start, end)
        months = _months_between(start.date(), last_day)
        parts = await asyncio.gather(
            *(
                self._once(("k", symbol, y, m), lambda y=y, m=m: self._kline_month(symbol, y, m))
                for y, m in months
            )
        )
        frames = [f for part in parts for f in part]
        archived = max((f["ts"].max() for f in frames), default=None)
        gap = start if archived is None else max(start, archived + timedelta(minutes=1))
        if gap < end:
            frames.append(await self._recent_klines(symbol, gap, end))
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise DataNotAvailable(f"Sin datos para {symbol} entre {start} y {end}")
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        df = df[(df["ts"] >= start) & (df["ts"] < end)]
        if df.empty:
            raise DataNotAvailable(f"Sin datos para {symbol} entre {start} y {end}")
        return df.reset_index(drop=True)

    async def _funding_month(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        tag = f"{year:04d}-{month:02d}"
        url = f"{BASE_URLS['futures']}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{tag}.zip"
        try:
            return parse_funding_csv(await self._zip(url))
        except DataNotAvailable:
            pass
        # Mes en curso: se pide el mes entero a la API.
        cursor = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
        end_ms = int(self.now.timestamp() * 1000)
        rows: list[dict] = []
        while cursor < end_ms:
            batch = await self._json(
                FUNDING_API,
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            rows.extend(batch)
            if len(batch) < 1000:
                break
            cursor = int(batch[-1]["fundingTime"]) + 1
        if not rows:
            return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": []})
        df = pd.DataFrame(rows)
        return pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    df["fundingTime"].astype("int64"), unit="ms", utc=True
                ).dt.floor("min"),
                "rate": df["fundingRate"].astype("float64"),
            }
        )

    async def funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        start, end, last_day = _utc_range(start, end)
        frames = await asyncio.gather(
            *(
                self._once(("f", symbol, y, m), lambda y=y, m=m: self._funding_month(symbol, y, m))
                for y, m in _months_between(start.date(), last_day)
            )
        )
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        return df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)


def _base_row(bot: PionexBot) -> dict:
    return {
        "bot": bot.bot_id,
        "symbol": bot.binance_symbol,
        "direction": bot.direction,
        "mode": bot.mode,
        "start": bot.start.strftime("%Y-%m-%d %H:%M"),
        "days": round(bot.days, 1),
        "lower": bot.lower,
        "upper": bot.upper,
        "grids": bot.grids,
        "modified": bot.irregular,
        "liquidated": bot.liquidated,
        "real_fills": len(bot.fills),
        "real_net": round(bot.real_net, 2),
        "real_gross": round(bot.real_pnl, 2),
        "real_fees": round(-bot.real_position_fees, 2),
        "real_funding": round(bot.real_funding, 2),
    }


async def analyze(export: bytes, get: Get, progress: Progress | None = None) -> list[dict]:
    """Reconstruye y valida todos los grids de futuros de un export de Pionex.

    Devuelve una fila por bot con los importes reales, los simulados y la
    coincidencia. `progress` recibe {"done", "total", "symbol"} tras cada bot.
    """
    trades, positions = load_export(export)
    bots = detect_bots(trades, positions)
    loader = Loader(get)

    async def prices(bot: PionexBot) -> tuple[pd.DataFrame, pd.DataFrame]:
        start, end = sim_window(bot)
        return await asyncio.gather(
            loader.candles(bot.binance_symbol, start, end),
            loader.funding(bot.binance_symbol, start, end),
        )

    # Todas las descargas arrancan ya (el límite de concurrencia lo pone `get`);
    # las simulaciones van en orden según llegan sus datos.
    pending = [asyncio.ensure_future(prices(bot)) for bot in bots]
    rows = []
    for i, (bot, task) in enumerate(zip(bots, pending, strict=True)):
        row = _base_row(bot)
        try:
            candles, funding = await task
            v = compare_bot(bot, candles, funding)
        except DataNotAvailable:
            row["error"] = "no_data"
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)[:200]
        else:
            tolerance = max(0.05 * abs(bot.real_net), 1.0)
            row |= {
                "sim_fills": v.sim_fills,
                "match": round(v.fill_match * 100, 1),
                "sim_net": round(v.sim_net, 2),
                "sim_fees": round(v.result.fees_paid, 2),
                "sim_funding": round(v.result.funding_pnl, 2),
                "grid_profit": round(v.result.grid_profit, 2),
                "round_trips": v.result.round_trips,
                "net_diff": round(v.net_diff, 2),
                "net_diff_pct": (
                    round(100 * v.net_diff / abs(bot.real_net), 1) if bot.real_net else None
                ),
                "within": bool(abs(v.net_diff) <= tolerance),
            }
        rows.append(row)
        if progress is not None:
            progress({"done": i + 1, "total": len(bots), "symbol": bot.binance_symbol})
    return rows


def shareable(rows: list[dict]) -> list[dict]:
    """Versión anonimizada para la página de validación: sin importes en USDT,
    sin identificadores y sin fechas exactas (solo el año)."""
    out = []
    for r in rows:
        if "match" not in r:
            continue
        out.append(
            {
                "symbol": r["symbol"].removesuffix("USDT"),
                "direction": r["direction"],
                "year": int(r["start"][:4]),
                "days": r["days"],
                "grids": r["grids"],
                "fills": r["real_fills"],
                "match": r["match"],
                "net_diff_pct": r["net_diff_pct"],
                "within": r["within"],
                "modified": r["modified"],
                "liquidated": r["liquidated"],
            }
        )
    return out


async def robustness(params: dict, get: Get, progress: Progress | None = None) -> dict:
    """Prueba de robustez del simulador web: `params` llega tal cual del formulario.

    Claves: symbol, exchange, direction, mode, below, above (en %), grids,
    leverage, days (duración de cada ventana) y months (histórico a recorrer).
    Devuelve {"rows": [...], "summary": {...}} o {"error": código}.
    """
    symbol = str(params["symbol"]).upper().strip().removesuffix("/USDT")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    if not symbol.isalnum() or len(symbol) > 24:
        return {"error": "symbol"}
    maker, taker = fees_for(str(params["exchange"]), futures=True)
    days = float(params["days"])
    months = int(params["months"])
    try:
        tpl = Template(
            direction=params["direction"],
            mode=params["mode"],
            below=float(params["below"]) / 100,
            above=float(params["above"]) / 100,
            grids=int(params["grids"]),
            leverage=float(params["leverage"]),
            maker=maker,
            taker=taker,
        )
        tpl.config(100.0)  # valida rango, grids y apalancamiento
    except (ValueError, KeyError):
        return {"error": "config"}
    if not (1 <= days <= 180 and 1 <= months <= 36 and tpl.below < 1):
        return {"error": "config"}

    end = pd.Timestamp(datetime.now(UTC)).floor("min")
    start = end - pd.Timedelta(days=30.4 * months)
    loader = Loader(get)
    try:
        candles, funding = await asyncio.gather(
            loader.candles(symbol, start, end), loader.funding(symbol, start, end)
        )
    except DataNotAvailable:
        return {"error": "no_data", "symbol": symbol}
    if progress is not None:
        progress({"stage": "sim"})
    rows = run_windows(candles, funding, tpl, days, progress=progress)
    if not rows:
        return {"error": "too_short", "symbol": symbol}
    return {
        "symbol": symbol,
        "from": str(candles["ts"].iloc[0])[:10],
        "to": str(candles["ts"].iloc[-1])[:10],
        "rows": rows,
        "summary": summarize(rows),
    }
