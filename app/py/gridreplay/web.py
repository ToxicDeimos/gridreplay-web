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
from dataclasses import replace
from urllib.parse import urlencode

import numpy as np
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
from gridreplay.sim import GridConfig, run_grid
from gridreplay.validate import bot_config, compare_bot, sim_window

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


# ---------------------------------------------------------------- «¿y si hubieras seguido al oráculo?»
#
# Para cada bot real se simula el bot que habría sugerido el oráculo ese día, con el mismo
# par, el mismo arranque y el mismo tamaño de posición (nocional), durante los 30 días para
# los que está pensado y validado (no la duración del bot real: con bots de 1–2 días el rango
# del oráculo, ancho, casi no opera y la comparación no diría nada):
#   * dirección: la que dio el oráculo en el arranque más reciente (cada 3 días) anterior al
#     bot, calculada sin mirar al futuro (scripts/oracle_history.py → oracle-history.json);
#   * rango: perfil equilibrado, ±2·vol·√30 con la volatilidad de los 14 días previos;
#     grids como en la web (site/oracle_template.html → sized);
#   * 2x de apalancamiento y SL/TP un 0,1 % fuera del rango (lo que recomienda la web).
# Se cierra a mercado a los 30 días (o antes, si salta un stop o se liquida; o hoy, si aún
# no han pasado: entonces se marca como en curso).

ORACLE_HISTORY_URL = "/app/oracle-history.json"
ORACLE_K = 2.0
ORACLE_LEVERAGE = 2.0
ORACLE_DAYS = 30
STOP_BUF = 0.001
DIRS = {"L": "long", "S": "short", "N": "neutral"}


async def load_history(get: Get) -> dict:
    """Historial del oráculo (vacío si no se puede descargar: el análisis sigue sin él)."""
    try:
        status, body = await get(ORACLE_HISTORY_URL)
        return json.loads(body) if status == 200 else {}
    except (OSError, ValueError):
        return {}


def oracle_direction(history: dict, symbol: str, ts: pd.Timestamp) -> str | None:
    """Dirección del oráculo vigente en `ts` (último arranque ≤ ts), o None si no hay dato."""
    p = history.get("pairs", {}).get(symbol) or history.get("pairs", {}).get("1000" + symbol)
    if not p:
        return None
    step = history.get("step_days", 3)
    start = pd.Timestamp(p["start"], tz="UTC")
    k = int(np.floor((ts - start) / pd.Timedelta(days=step)))
    if k < 0 or k >= len(p["dirs"]):  # cada arranque vale hasta el siguiente (3 días)
        return None
    return DIRS.get(p["dirs"][k])


def daily_vol(candles: pd.DataFrame) -> float | None:
    """Volatilidad diaria con las velas horarias de los 14 días previos (como el oráculo)."""
    if candles.empty:
        return None
    h = candles.set_index("ts")["close"].astype(float).resample("1h").last().dropna()
    lr = np.log(h).diff().dropna().to_numpy()
    if len(lr) < 24 * 7:
        return None
    return float(np.std(lr[-14 * 24:], ddof=1) * np.sqrt(24))


def oracle_config(bot: PionexBot, direction: str, vol: float) -> GridConfig:
    half = min(max(ORACLE_K * vol * 30**0.5, 0.02), 0.7)
    grids = round(min(max(2 * half / (0.5 * vol), 8), 200))
    notional = bot_config(bot).investment  # mismo tamaño de posición que el bot real
    cfg = Template(direction, half, half, grids, ORACLE_LEVERAGE, maker=bot.maker_fee, taker=bot.taker_fee,
                   investment=notional / ORACLE_LEVERAGE).config(bot.entry_price)
    lo, hi = cfg.lower * (1 - STOP_BUF), cfg.upper * (1 + STOP_BUF)
    sl, tp = (hi, lo) if direction == "short" else (lo, hi)
    return replace(cfg, stop_loss=sl, take_profit=tp, entry_price=bot.entry_price)


def oracle_window(bot: PionexBot, now: datetime) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(inicio de los 14 días previos, arranque, fin de los 30 días o ahora)."""
    start = sim_window(bot)[0]
    end = min(start + pd.Timedelta(days=ORACLE_DAYS), pd.Timestamp(now).floor("min"))
    return start - pd.Timedelta(days=15), start, end


def oracle_bot(bot: PionexBot, history: dict, candles: pd.DataFrame, funding: pd.DataFrame,
               now: datetime) -> dict:
    """Resultado del bot sugerido por el oráculo, o {'oracle': None, 'oracle_why': motivo}.

    `candles` cubre desde 15 días antes del arranque hasta el final de los 30 días."""
    direction = oracle_direction(history, bot.binance_symbol, bot.start)
    if direction is None:
        return {"oracle": None, "oracle_why": "no_signal"}
    _, start, end = oracle_window(bot, now)
    before = candles[candles["ts"] < start]
    after = candles[(candles["ts"] >= start) & (candles["ts"] < end)].reset_index(drop=True)
    vol = daily_vol(before)
    if vol is None or vol <= 0 or after.empty:
        return {"oracle": None, "oracle_why": "no_data"}
    cfg = oracle_config(bot, direction, vol)
    fund = funding[(funding["ts"] >= start) & (funding["ts"] < end)] if not funding.empty else funding
    res = run_grid(after, cfg, funding=fund, exit_price=float(after["close"].iloc[-1]))
    return {
        "oracle": {
            "direction": direction,
            "lower": round(cfg.lower, 8),
            "upper": round(cfg.upper, 8),
            "grids": cfg.grids,
            "net": round(res.total_pnl, 2),
            "stopped": res.stopped,
            "days": round((end - start) / pd.Timedelta(days=1), 1),
            "partial": bool(end - start < pd.Timedelta(days=ORACLE_DAYS)),
        }
    }


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
    history = await load_history(get)

    async def prices(bot: PionexBot) -> tuple[pd.DataFrame, ...]:
        start, end = sim_window(bot)
        o_from, o_start, o_end = oracle_window(bot, loader.now)
        empty = asyncio.sleep(0, pd.DataFrame())
        return await asyncio.gather(
            loader.candles(bot.binance_symbol, start, end),
            loader.funding(bot.binance_symbol, start, end),
            # bot del oráculo: 14 días previos (rango) + sus 30 días
            loader.candles(bot.binance_symbol, o_from, o_end) if history else empty,
            loader.funding(bot.binance_symbol, o_start, o_end) if history else asyncio.sleep(0, pd.DataFrame()),
        )

    # Todas las descargas arrancan ya (el límite de concurrencia lo pone `get`);
    # las simulaciones van en orden según llegan sus datos.
    pending = [asyncio.ensure_future(prices(bot)) for bot in bots]
    rows = []
    for i, (bot, task) in enumerate(zip(bots, pending, strict=True)):
        row = _base_row(bot)
        try:
            candles, funding, o_candles, o_funding = await task
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
            try:
                row |= (oracle_bot(bot, history, o_candles, o_funding, loader.now) if history
                        else {"oracle": None, "oracle_why": "no_history"})
            except (OSError, ValueError) as exc:
                row |= {"oracle": None, "oracle_why": str(exc)[:120]}
        rows.append(row)
        if progress is not None:
            progress({"done": i + 1, "total": len(bots), "symbol": bot.binance_symbol})
    return rows


def excluded(export: bytes) -> dict:
    """Futuros del export que el analizador no reconstruye (bots con menos de 20 órdenes
    límite, operaciones manuales). Sin ellos el resultado del analizador sale mejor que el
    real: se enseñan aparte y en el total real de la cuenta."""
    trades, positions = load_export(export)
    ids = {b.bot_id for b in detect_bots(trades, positions)}
    out = positions[~positions["tax_id"].isin(ids)]
    if out.empty:
        return {"groups": 0, "positions": 0, "net": 0.0, "fees": 0.0, "funding": 0.0}
    net = out["pnl"] + out["fee"] + out["funding_fee"]
    return {
        "groups": int(out["tax_id"].nunique()),
        "positions": int(len(out)),
        "net": round(float(net.sum()), 2),
        "fees": round(float(-out["fee"].sum()), 2),
        "funding": round(float(out["funding_fee"].sum()), 2),
    }


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
