"""Importador del export de transacciones de Pionex.

El export (zip con `raw-trading-details.csv` y `position_futures.csv`) no trae
la configuración de los bots, pero sí cada orden ejecutada con un `tax_id` por
bot. A partir de las órdenes se reconstruye el grid:

* Niveles: las órdenes límite caen en una rejilla regular (aritmética o
  geométrica); se ajusta paso y origen por mínimos cuadrados.
* Cantidad por nivel: la moda de las cantidades de las órdenes límite.
* Dirección: la orden a mercado del arranque (compra → long, venta → short;
  sin ella → neutral).
* Límites: el número de niveles comprados/vendidos en la orden inicial fija el
  límite superior (long) o inferior (short); el otro límite es el nivel más
  extremo observado (los niveles nunca tocados no aparecen en el export).
* Fin: la orden a mercado final cierra la posición.

Ojo: la fecha de cada orden en el export es la de *creación*, no la de
ejecución, así que la comparación se hace por nivel y no por hora.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

MARKET_TYPES = {"IOC", "MARKET_QTY", "MARKET_AMT"}
START_WINDOW = pd.Timedelta(seconds=5)


@dataclass
class PionexBot:
    bot_id: str
    symbol: str
    direction: Literal["long", "short", "neutral"]
    mode: Literal["arithmetic", "geometric"]
    start: pd.Timestamp
    end: pd.Timestamp
    entry_price: float
    exit_price: float | None
    qty: float
    origin: float  # nivel de índice 0 (o su logaritmo, si es geométrico)
    step: float  # separación (o ratio logarítmico, si es geométrico)
    k_lower: int
    k_upper: int
    maker_fee: float
    taker_fee: float  # apertura a mercado
    exit_fee: float | None  # cierre (≈1 % si fue una liquidación)
    fills: pd.DataFrame  # órdenes límite del grid: ts, side, price, level
    real_gross: float  # Σ ventas − Σ compras de todas las órdenes del bot
    real_fees: float
    real_pnl: float  # columnas del fichero de posiciones
    real_position_fees: float
    real_funding: float
    off_grid: float  # fracción de órdenes límite fuera de la rejilla
    mixed_qty: float  # fracción de órdenes límite con otra cantidad (bot modificado)

    @property
    def irregular(self) -> bool:
        """Bot modificado en marcha (inversión añadida, rango cambiado…): no comparable."""
        return self.off_grid > 0.05 or self.mixed_qty > 0.1

    @property
    def liquidated(self) -> bool:
        """El cierre pagó la comisión de liquidación (muy superior a la taker)."""
        return self.exit_fee is not None and self.exit_fee > 0.005

    @property
    def binance_symbol(self) -> str:
        return self.symbol.replace("_USDT_PERP", "USDT").replace("_", "")

    @property
    def grids(self) -> int:
        return self.k_upper - self.k_lower

    def level(self, k: int | np.ndarray) -> float | np.ndarray:
        x = self.origin + np.asarray(k) * self.step
        return np.exp(x) if self.mode == "geometric" else x

    @property
    def lower(self) -> float:
        return float(self.level(self.k_lower))

    @property
    def upper(self) -> float:
        return float(self.level(self.k_upper))

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400

    @property
    def real_net(self) -> float:
        return self.real_pnl + self.real_position_fees + self.real_funding


def load_export(path: str | Path | bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee órdenes de futuros ejecutadas y posiciones desde el zip (ruta o bytes) o
    desde su carpeta descomprimida."""
    source = io.BytesIO(path) if isinstance(path, bytes) else Path(path)

    def read(name: str) -> pd.DataFrame:
        if isinstance(source, Path) and source.is_dir():
            return pd.read_csv(source / name)
        with zipfile.ZipFile(source) as zf:
            return pd.read_csv(io.BytesIO(zf.read(name)))

    trades = read("raw-trading-details.csv").rename(columns={"date(UTC+0)": "ts"})
    trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
    trades = trades[(trades["market_type"] != "Spot") & (trades["state"] == "FILLED")]

    positions = read("position_futures.csv")
    positions["tax_id"] = "s_" + positions["position_id"].str.split("_").str[1]
    return trades.reset_index(drop=True), positions


def _fit_grid(prices: np.ndarray, geometric: bool) -> tuple[float, float, np.ndarray, float]:
    """Ajusta origen y paso; devuelve (origen, paso, índices, error relativo medio)."""
    x = np.log(prices) if geometric else prices
    ux = np.unique(x)
    diffs = np.diff(ux)
    if diffs.size == 0:
        return float(ux[0]), 0.0, np.zeros(len(x), dtype=int), np.inf
    step = float(np.median(diffs))
    origin = float(ux[0])
    for _ in range(3):
        k = np.round((ux - origin) / step)
        if np.ptp(k) == 0:
            break
        step, origin = np.polyfit(k, ux, 1)
    k_all = np.round((x - origin) / step).astype(int)
    rel_err = np.abs(x - (origin + k_all * step)) / step
    return float(origin), float(step), k_all, rel_err


def _reconstruct(bot_id: str, g: pd.DataFrame, positions: pd.DataFrame) -> PionexBot | None:
    g = g.sort_values("ts", kind="stable")
    limit = g[g["order_type"] == "LIMIT"]
    if limit.empty:
        return None
    qty = float(limit["executed_qty"].round(10).mode().iloc[0])
    on_qty = limit[np.isclose(limit["executed_qty"], qty, rtol=1e-6)]
    prices = on_qty["price"].to_numpy(dtype=float)
    if len(np.unique(prices)) < 3:
        return None

    fits = {}
    for mode, geometric in (("arithmetic", False), ("geometric", True)):
        origin, step, k, rel_err = _fit_grid(prices, geometric)
        if step > 0:
            fits[mode] = (origin, step, k, rel_err)
    if not fits:
        return None
    mode = min(fits, key=lambda m: float(np.median(fits[m][3])))
    origin, step, k, rel_err = fits[mode]
    on_grid = rel_err < 0.1
    off_grid = 1 - on_grid.mean()

    start, end = g["ts"].iloc[0], g["ts"].iloc[-1]
    market = g[g["order_type"].isin(MARKET_TYPES)]
    # Apertura y cierre mueven varios niveles de golpe. Desde 2026 son órdenes a
    # mercado (IOC); en 2025 Pionex abría con una orden límite grande.
    bulk = g[g["executed_qty"] > 1.5 * qty]
    init = bulk[bulk["ts"] <= start + START_WINDOW]
    final = bulk[bulk["ts"] > start + START_WINDOW]

    def to_k(price: float) -> float:
        x = np.log(price) if mode == "geometric" else price
        return (x - origin) / step

    k_obs = k[on_grid]
    k_min, k_max = int(k_obs.min()), int(k_obs.max())
    if not init.empty:
        first = init.iloc[0]
        direction: Literal["long", "short", "neutral"] = (
            "long" if first["side"] == "BUY" else "short"
        )
        entry_price = float(first["price"])
        empty = int(np.round(to_k(entry_price)))
        n_init = int(round(init["executed_qty"].sum() / qty))
        if direction == "long":
            k_lower, k_upper = min(k_min, empty), max(k_max, empty + n_init)
        else:
            k_lower, k_upper = min(k_min, empty - n_init), max(k_max, empty)
    else:
        direction = "neutral"
        # Nivel vacío: el hueco entre la compra más alta y la venta más baja iniciales.
        first_orders = on_qty[on_qty["ts"] <= start + START_WINDOW]
        buys = first_orders.loc[first_orders["side"] == "BUY", "price"]
        sells = first_orders.loc[first_orders["side"] == "SELL", "price"]
        if buys.empty or sells.empty:
            return None
        empty = int(np.round((to_k(buys.max()) + to_k(sells.min())) / 2))
        x_empty = origin + empty * step
        entry_price = float(np.exp(x_empty) if mode == "geometric" else x_empty)
        k_lower, k_upper = k_min, k_max

    fills = pd.DataFrame(
        {
            "ts": on_qty["ts"].to_numpy()[on_grid],
            "side": on_qty["side"].str.lower().to_numpy()[on_grid],
            "price": prices[on_grid],
            "level": k_obs - k_lower,
        }
    )
    sign = np.where(g["side"] == "SELL", 1.0, -1.0)
    pos = positions[positions["tax_id"] == bot_id]

    def fee_rate(orders: pd.DataFrame) -> float:
        return float((orders["fee"] / orders["amount"]).median())

    maker = fee_rate(limit)
    opening = init if not init.empty else market
    taker = fee_rate(opening) if not opening.empty else maker
    exit_fee = fee_rate(final.tail(1)) if not final.empty else None

    return PionexBot(
        bot_id=bot_id,
        symbol=str(g["symbol"].iloc[0]),
        direction=direction,
        mode=mode,
        start=start,
        end=end,
        entry_price=entry_price,
        exit_price=float(final["price"].iloc[-1]) if not final.empty else None,
        qty=qty,
        origin=origin,
        step=step,
        k_lower=k_lower,
        k_upper=k_upper,
        maker_fee=maker,
        taker_fee=taker,
        exit_fee=exit_fee,
        fills=fills,
        real_gross=float((sign * g["amount"]).sum()),
        real_fees=float(g["fee"].sum()),
        real_pnl=float(pos["pnl"].sum()),
        real_position_fees=float(pos["fee"].sum()),
        real_funding=float(pos["funding_fee"].sum()),
        off_grid=float(off_grid),
        mixed_qty=float(1 - len(on_qty) / len(limit)),
    )


def detect_bots(
    trades: pd.DataFrame, positions: pd.DataFrame, min_fills: int = 20
) -> list[PionexBot]:
    """Reconstruye los grids de futuros con al menos `min_fills` órdenes límite."""
    bots = []
    for bot_id, g in trades.groupby("tax_id"):
        if (g["order_type"] == "LIMIT").sum() < min_fills:
            continue
        bot = _reconstruct(str(bot_id), g, positions)
        if bot is not None:
            bots.append(bot)
    return sorted(bots, key=lambda b: len(b.fills), reverse=True)
