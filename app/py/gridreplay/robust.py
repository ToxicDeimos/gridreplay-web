"""Prueba de robustez: la misma configuración de grid en muchos periodos.

Un backtest de un solo periodo dice poco: el mismo grid puede ganar en un mes
lateral y liquidarse al siguiente. Aquí la configuración se define *relativa al
precio de entrada* (rango −x % / +y %), se lanza al principio de cada ventana
de N días a lo largo del histórico y se resume la distribución de resultados:
cuántas veces gana, cuánto pierde en los peores casos y cuántas se liquida.

Las ventanas se solapan (arranques cada `step`) para tener muestras suficientes
sin necesitar años de datos; por eso no son independientes entre sí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from gridreplay.sim import GridConfig, run_grid

MAX_WINDOWS = 100


@dataclass(frozen=True)
class Template:
    """Configuración de grid relativa al precio de entrada."""

    direction: Literal["long", "short", "neutral"]
    below: float  # límite inferior = entrada × (1 − below)
    above: float  # límite superior = entrada × (1 + above)
    grids: int
    leverage: float
    mode: Literal["arithmetic", "geometric"] = "arithmetic"
    maker: float = 0.0002
    taker: float = 0.0005
    investment: float = 1000.0

    def config(self, price: float) -> GridConfig:
        return GridConfig(
            lower=price * (1 - self.below),
            upper=price * (1 + self.above),
            grids=self.grids,
            investment=self.investment,
            mode=self.mode,
            fee_rate=self.maker,
            taker_fee=self.taker,
            direction=self.direction,
            leverage=self.leverage,
        )


def window_starts(
    first: pd.Timestamp, last: pd.Timestamp, days: float, max_windows: int = MAX_WINDOWS
) -> list[pd.Timestamp]:
    """Arranques de ventanas de `days` días dentro de [first, last), a intervalos
    de al menos un día y como mucho `max_windows`, con la última pegada al final."""
    length = pd.Timedelta(days=days)
    span = last - first - length
    if span < pd.Timedelta(0):
        return []
    step = max(pd.Timedelta(days=1), span / max(max_windows - 1, 1))
    n = int(span / step) + 1
    latest = (last - length).floor("min")
    return [latest - i * step for i in range(n)][::-1]


def run_windows(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None,
    tpl: Template,
    days: float,
    max_windows: int = MAX_WINDOWS,
    progress=None,
) -> list[dict]:
    """Simula `tpl` en cada ventana; una fila por ventana con su resultado."""
    ts = pd.DatetimeIndex(candles["ts"])
    starts = window_starts(ts[0], ts[-1] + pd.Timedelta(minutes=1), days, max_windows)
    length = pd.Timedelta(days=days)
    rows = []
    for i, start in enumerate(starts):
        a, b = ts.searchsorted(start), ts.searchsorted(start + length)
        window = candles.iloc[a:b]
        if len(window) < 2:
            continue
        price = float(window["open"].iloc[0])
        fund = None
        if funding is not None and not funding.empty:
            fund = funding[(funding["ts"] >= start) & (funding["ts"] < start + length)]
        r = run_grid(window, tpl.config(price), funding=fund)
        inv = tpl.investment
        rows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "price": price,
                "ret": round(100 * r.total_return, 2),
                "hold": round(100 * (float(window["close"].iloc[-1]) / price - 1), 2),
                "grid": round(100 * r.grid_profit / inv, 2),
                "fees": round(100 * r.fees_paid / inv, 2),
                "funding": round(100 * r.funding_pnl / inv, 2),
                "dd": round(100 * r.max_drawdown, 2),
                "in_range": round(100 * r.time_in_range, 1),
                "trips": r.round_trips,
                "liquidated": r.stopped == "liquidation",
            }
        )
        if progress is not None:
            progress({"done": i + 1, "total": len(starts)})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Distribución de resultados de todas las ventanas."""
    if not rows:
        return {"windows": 0}
    ret = np.array([r["ret"] for r in rows])
    hold = np.array([r["hold"] for r in rows])
    p10, p50, p90 = np.percentile(ret, [10, 50, 90])
    return {
        "windows": len(rows),
        "win_rate": round(100 * float(np.mean(ret > 0)), 1),
        "median": round(float(p50), 2),
        "p10": round(float(p10), 2),
        "p90": round(float(p90), 2),
        "worst": round(float(ret.min()), 2),
        "best": round(float(ret.max()), 2),
        "liquidations": int(sum(r["liquidated"] for r in rows)),
        "median_dd": round(float(np.median([r["dd"] for r in rows])), 2),
        "median_in_range": round(float(np.median([r["in_range"] for r in rows])), 1),
        "median_fees": round(float(np.median([r["fees"] for r in rows])), 2),
        "beats_hold": round(100 * float(np.mean(ret > hold)), 1),
    }
