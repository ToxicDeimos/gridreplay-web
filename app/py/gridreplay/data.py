"""Velas de 1 minuto y funding desde data.binance.vision, cacheados en Parquet.

Estructura de la caché:
    <cache>/<market>/<SYMBOL>/1m/YYYY-MM.parquet         meses completos
    <cache>/<market>/<SYMBOL>/1m/YYYY-MM-DD.parquet      días del mes en curso
    <cache>/futures/<SYMBOL>/funding/YYYY-MM.parquet     funding (meses cerrados)

`market` es "spot" o "futures" (perpetuos USDⓈ-M). Cada descarga se verifica
contra el SHA-256 que publica Binance. Las últimas horas, que todavía no están
en el archivo diario, se piden a la API pública de klines y no se cachean.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

Market = Literal["spot", "futures"]

BASE_URLS: dict[str, str] = {
    "spot": "https://data.binance.vision/data/spot",
    "futures": "https://data.binance.vision/data/futures/um",
}
FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINE_APIS: dict[str, str] = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}
# La API solo rellena lo que el archivo aún no publica (sea cual sea el retraso:
# si el archivo diario va días por detrás, cortar ahí truncaría el periodo).
DEFAULT_CACHE = Path("data/candles")
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# Desde 2025-01-01 los ficheros spot traen timestamps en microsegundos.
_MICROSECOND_THRESHOLD = 10**14


class DataNotAvailable(Exception):
    """Binance todavía no publica ese periodo (o el símbolo no existe)."""


def _to_utc(ms_or_us: pd.Series) -> pd.Series:
    values = ms_or_us.astype("int64")
    unit = "us" if values.max() > _MICROSECOND_THRESHOLD else "ms"
    return pd.to_datetime(values, unit=unit, utc=True)


def parse_kline_csv(raw: bytes) -> pd.DataFrame:
    """Convierte el CSV de klines de Binance en un DataFrame normalizado."""
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=range(6))
    # Los ficheros de futuros incluyen cabecera: la detectamos y la descartamos.
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df.columns = COLUMNS
    out = pd.DataFrame({"ts": _to_utc(df["ts"])})
    for col in COLUMNS[1:]:
        out[col] = df[col].astype("float64").to_numpy()
    return out.reset_index(drop=True)


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """CSV de funding de Binance → columnas ts, rate."""
    df = pd.read_csv(io.BytesIO(raw))
    return pd.DataFrame(
        {
            "ts": _to_utc(df["calc_time"]).dt.floor("min"),
            "rate": df["last_funding_rate"].astype("float64"),
        }
    )


def _download_zip(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60)
    if resp.status_code == 404:
        raise DataNotAvailable(url)
    resp.raise_for_status()
    payload = resp.content

    check = session.get(url + ".CHECKSUM", timeout=30)
    check.raise_for_status()
    expected = check.text.split()[0].strip().lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise OSError(f"Checksum incorrecto para {url}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return zf.read(zf.namelist()[0])


def _cached(path: Path, fetch) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _month_frames(
    session: requests.Session, market: Market, symbol: str, year: int, month: int, cache: Path
) -> list[pd.DataFrame]:
    base = BASE_URLS[market]
    folder = cache / market / symbol / "1m"
    tag = f"{year:04d}-{month:02d}"
    url = f"{base}/monthly/klines/{symbol}/1m/{symbol}-1m-{tag}.zip"
    try:
        return [
            _cached(folder / f"{tag}.parquet", lambda: parse_kline_csv(_download_zip(session, url)))
        ]
    except DataNotAvailable:
        pass

    # Mes en curso: Binance solo publica ficheros diarios (con un día de retraso).
    frames = []
    day = date(year, month, 1)
    today = datetime.now(UTC).date()
    while day.month == month and day < today:
        url = f"{base}/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
        try:
            frames.append(
                _cached(
                    folder / f"{day.isoformat()}.parquet",
                    lambda url=url: parse_kline_csv(_download_zip(session, url)),
                )
            )
        except DataNotAvailable:
            break
        day += timedelta(days=1)
    return frames


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _utc_range(start: datetime, end: datetime) -> tuple[datetime, datetime, date]:
    start = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
    end = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
    # `end` es exclusivo: 2025-10-01 00:00 no necesita descargar octubre.
    return start, end, (end - timedelta(microseconds=1)).date()


def _recent_klines(
    session: requests.Session, market: Market, symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    rows: list[list] = []
    cursor = int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    while cursor < stop:
        resp = session.get(
            KLINE_APIS[market],
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": stop - 1,
                "limit": 1500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
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
    # La vela del minuto en curso todavía no está cerrada.
    return out[out["ts"] + pd.Timedelta(minutes=1) <= datetime.now(UTC)]


def load_candles(
    symbol: str,
    start: datetime,
    end: datetime,
    cache: Path = DEFAULT_CACHE,
    market: Market = "spot",
) -> pd.DataFrame:
    """Devuelve las velas de 1m en [start, end), descargando lo que falte."""
    symbol = symbol.upper()
    start, end, last_day = _utc_range(start, end)

    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for year, month in _months_between(start.date(), last_day):
            frames.extend(_month_frames(session, market, symbol, year, month, cache))
        archived = max((f["ts"].max() for f in frames), default=None)
        gap = start if archived is None else max(start, archived + timedelta(minutes=1))
        if gap < end:
            frames.append(_recent_klines(session, market, symbol, gap, end))
        frames = [f for f in frames if not f.empty]

    if not frames:
        raise DataNotAvailable(f"Sin datos para {symbol} ({market}) entre {start} y {end}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df[(df["ts"] >= start) & (df["ts"] < end)]
    if df.empty:
        raise DataNotAvailable(f"Sin datos para {symbol} ({market}) entre {start} y {end}")
    return df.reset_index(drop=True)


def _funding_from_api(session: requests.Session, symbol: str, start: datetime, end: datetime):
    rows: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        resp = session.get(
            FUNDING_API,
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": []})
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True).dt.floor(
                "min"
            ),
            "rate": df["fundingRate"].astype("float64"),
        }
    )


def load_funding(
    symbol: str, start: datetime, end: datetime, cache: Path = DEFAULT_CACHE
) -> pd.DataFrame:
    """Histórico de funding (ts, rate) del perpetuo en [start, end)."""
    symbol = symbol.upper()
    start, end, last_day = _utc_range(start, end)
    folder = cache / "futures" / symbol / "funding"
    base = BASE_URLS["futures"]

    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for year, month in _months_between(start.date(), last_day):
            tag = f"{year:04d}-{month:02d}"
            url = f"{base}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{tag}.zip"
            try:
                frames.append(
                    _cached(
                        folder / f"{tag}.parquet",
                        lambda url=url: parse_funding_csv(_download_zip(session, url)),
                    )
                )
            except DataNotAvailable:
                # Mes en curso: no hay fichero mensual, se pide a la API (sin caché).
                month_start = datetime(year, month, 1, tzinfo=UTC)
                frames.append(_funding_from_api(session, symbol, max(start, month_start), end))

    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    return df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
