"""Simulador de grids (spot y futuros long / short / neutral) sobre velas OHLC.

Modelo
------
El rango [lower, upper] se divide en `grids` celdas (grids + 1 niveles). Cada
celda i opera siempre la misma cantidad `qty[i]` y está en uno de dos estados:

* abajo:  orden de compra en levels[i];
* arriba: orden de venta en levels[i + 1].

Al arrancar, el nivel más cercano al precio de entrada queda vacío (en empate,
el de arriba): las celdas por encima empiezan "arriba" (venden) y las de debajo
"abajo" (compran). Hay `grids` órdenes en `grids` niveles distintos, igual que
en Pionex. Invariante: toda compra pendiente está por debajo del precio y toda
venta pendiente por encima.

La dirección decide qué posición representa cada estado:

* long:    abajo = sin posición, arriba = +qty (se compra a mercado al arrancar);
* short:   abajo = -qty, arriba = sin posición (se vende a mercado al arrancar);
* neutral: celdas bajo la entrada se comportan como long y las de encima como
  short; arranca sin posición.

Un ciclo completo (round trip) es compra + venta de una celda; su beneficio
(qty × separación) se contabiliza al cerrarse: en la venta para celdas long y
en la recompra para celdas short.

Supuestos (verificados contra exports reales de Pionex donde se indica)
----------------------------------------------------------------------
* Cantidad igual por nivel (Pionex): qty = investment × leverage / Σ niveles de
  compra. Alternativa `equal_quote`: mismo importe en cuota por celda.
* Posición inicial a mercado con comisión taker; órdenes del grid con comisión
  maker (`fee_rate`). Pionex futuros: 0,02 % maker, 0,05 % taker.
* Recorrido intravela: una vela alcista va open → low → high → close, y una
  bajista open → high → low → close. El hueco entre el cierre anterior y la
  apertura también se recorre.
* Una orden límite se ejecuta en cuanto el precio toca su nivel, al precio del
  nivel.
* Liquidación (margen aislado = inversión): cuando equity ≤ margen de
  mantenimiento × nocional. Se cierra al precio de liquidación y se pierde la
  inversión. El funding no se tiene en cuenta para la liquidación.
* Funding: en cada liquidación de funding, pago = −posición × precio × tasa.
* Stop-loss y take-profit cierran toda la posición al precio de disparo
  (taker) y detienen el bot. Long/neutral: SL < lower y TP > upper; short: al
  revés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Mode = Literal["arithmetic", "geometric"]
Direction = Literal["long", "short", "neutral"]
QtyMode = Literal["equal_base", "equal_quote"]
MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass(frozen=True)
class GridConfig:
    lower: float
    upper: float
    grids: int
    investment: float
    mode: Mode = "arithmetic"
    fee_rate: float = 0.0005
    taker_fee: float | None = None
    direction: Direction = "long"
    leverage: float = 1.0
    qty_mode: QtyMode = "equal_base"
    qty: float | None = None
    entry_price: float | None = None
    maintenance_margin: float = 0.004
    stop_loss: float | None = None
    take_profit: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.lower < self.upper:
            raise ValueError("Se necesita 0 < lower < upper")
        if self.grids < 1:
            raise ValueError("grids debe ser >= 1")
        if self.investment <= 0:
            raise ValueError("investment debe ser > 0")
        if not 0 <= self.fee_rate < 0.1 or not 0 <= self.taker < 0.1:
            raise ValueError("comisión fuera de rango")
        if self.mode not in ("arithmetic", "geometric"):
            raise ValueError("mode debe ser 'arithmetic' o 'geometric'")
        if self.direction not in ("long", "short", "neutral"):
            raise ValueError("direction debe ser 'long', 'short' o 'neutral'")
        if self.qty_mode not in ("equal_base", "equal_quote"):
            raise ValueError("qty_mode debe ser 'equal_base' o 'equal_quote'")
        if not 1 <= self.leverage <= 125:
            raise ValueError("leverage debe estar entre 1 y 125")
        if self.qty is not None and self.qty <= 0:
            raise ValueError("qty debe ser > 0")
        below, above = self._stops()
        if below is not None and not 0 < below[0] < self.lower:
            raise ValueError(f"{below[1]} debe estar por debajo de lower")
        if above is not None and above[0] <= self.upper:
            raise ValueError(f"{above[1]} debe estar por encima de upper")

    @property
    def taker(self) -> float:
        return self.fee_rate if self.taker_fee is None else self.taker_fee

    def _stops(self) -> tuple[tuple[float, str] | None, tuple[float, str] | None]:
        """(disparo por debajo del rango, disparo por encima), con su motivo."""
        sl = (self.stop_loss, "stop_loss") if self.stop_loss is not None else None
        tp = (self.take_profit, "take_profit") if self.take_profit is not None else None
        return (tp, sl) if self.direction == "short" else (sl, tp)

    def levels(self) -> np.ndarray:
        if self.mode == "arithmetic":
            return np.linspace(self.lower, self.upper, self.grids + 1)
        return np.geomspace(self.lower, self.upper, self.grids + 1)

    def quantities(self) -> np.ndarray:
        buy_levels = self.levels()[:-1]
        if self.qty is not None:
            return np.full(self.grids, self.qty)
        capital = self.investment * self.leverage
        if self.qty_mode == "equal_quote":
            return (capital / self.grids) / buy_levels
        return np.full(self.grids, capital / buy_levels.sum())


@dataclass
class GridResult:
    config: GridConfig
    start_price: float
    end_price: float
    final_equity: float
    final_position: float
    grid_profit: float
    fees_paid: float
    funding_pnl: float
    round_trips: int
    buy_hold_equity: float
    time_in_range: float
    range_exits: int
    max_drawdown: float
    minutes: int  # minutos activos (hasta el stop, si lo hubo)
    stopped: str | None
    stopped_at: pd.Timestamp | None
    equity: pd.Series = field(repr=False)
    trades: pd.DataFrame = field(repr=False)

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.config.investment

    @property
    def position_pnl(self) -> float:
        """Lo que no es grid ni funding: revalorización de la posición abierta."""
        return self.total_pnl - self.grid_profit - self.funding_pnl

    @property
    def total_return(self) -> float:
        return self.total_pnl / self.config.investment

    @property
    def buy_hold_return(self) -> float:
        return self.buy_hold_equity / self.config.investment - 1

    @property
    def grid_apr(self) -> float:
        """Beneficio del grid anualizado (simple), como lo muestran los exchanges."""
        if self.minutes == 0:
            return 0.0
        return self.grid_profit / self.config.investment * MINUTES_PER_YEAR / self.minutes


def _liquidation_price(investment: float, cash: float, pos: float, mm: float) -> float | None:
    """Precio al que equity = margen de mantenimiento, o None si no aplica."""
    equity_at_zero = investment + cash  # equity(p) = equity_at_zero + pos * p
    if pos > 0:
        p = -equity_at_zero / (pos * (1 - mm))
        return p if p > 0 else None
    if pos < 0:
        return max(equity_at_zero, 0.0) / (-pos * (1 + mm))
    return None


def run_grid(
    candles: pd.DataFrame,
    cfg: GridConfig,
    funding: pd.DataFrame | None = None,
    exit_price: float | None = None,
    exit_fee: float | None = None,
) -> GridResult:
    """Ejecuta el grid sobre `candles` (columnas ts, open, high, low, close).

    `funding` (columnas ts, rate) aplica pagos de funding. `exit_price` cierra la
    posición al final a ese precio, como al detener un bot, con comisión
    `exit_fee` (por defecto la taker).
    """
    if candles.empty:
        raise ValueError("No hay velas")

    levels = cfg.levels()
    buy_levels = levels[:-1]
    sell_levels = levels[1:]
    n = cfg.grids
    maker, taker = cfg.fee_rate, cfg.taker
    inv, mm = cfg.investment, cfg.maintenance_margin
    qty = cfg.quantities()

    stamps = pd.DatetimeIndex(candles["ts"])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    o = candles["open"].to_numpy(dtype="float64")
    h = candles["high"].to_numpy(dtype="float64")
    lo = candles["low"].to_numpy(dtype="float64")
    c = candles["close"].to_numpy(dtype="float64")
    m = len(c)

    p0 = float(cfg.entry_price if cfg.entry_price is not None else o[0])
    # Igual que el exchange, no se puede arrancar con el precio ya más allá de un stop.
    below, above = cfg._stops()
    for trigger, crossed in ((below, lambda t: p0 <= t), (above, lambda t: p0 >= t)):
        if trigger is not None and crossed(trigger[0]):
            raise ValueError(f"El precio de entrada {p0:g} ya ha superado el {trigger[1]}")
    distance = np.abs(levels - p0)
    empty = len(levels) - 1 - int(np.argmin(distance[::-1]))  # empate → nivel superior
    cells = np.arange(n)
    holding = cells >= empty
    if cfg.direction == "long":
        floor_pos = np.zeros(n)
    elif cfg.direction == "short":
        floor_pos = -qty
    else:
        floor_pos = np.where(cells >= empty, -qty, 0.0)
    short_cell = floor_pos < 0

    pos = float((floor_pos + qty * holding).sum())
    cash = 0.0  # flujo de caja acumulado; equity = inv + cash + pos × precio
    fees_paid = grid_profit = 0.0
    round_trips = 0
    trades: list[tuple] = []

    def close_position(price: float, fee_rate: float = taker) -> float:
        nonlocal cash, fees_paid, grid_profit, pos
        size = abs(pos)
        fee = size * price * fee_rate
        cash += pos * price - fee
        fees_paid += fee
        grid_profit -= fee
        pos = 0.0
        return size

    if pos != 0:
        fee = abs(pos) * p0 * taker
        cash -= pos * p0 + fee
        fees_paid += fee
        grid_profit -= fee
        trades.append((stamps[0], "buy" if pos > 0 else "sell", p0, abs(pos), -1, "init"))

    # Velas que no tocan ningún nivel ni stop no cambian el estado: se saltan
    # (salvo para comprobar la liquidación, que se hace en bloque).
    prev_close = np.concatenate(([p0], c[:-1]))
    span_lo = np.minimum(lo, prev_close)
    span_hi = np.maximum(h, prev_close)
    touches = np.searchsorted(levels, span_hi, "right") > np.searchsorted(levels, span_lo, "left")
    trigger_below, trigger_above = cfg._stops()
    if trigger_below is not None:
        touches |= span_lo <= trigger_below[0]
    if trigger_above is not None:
        touches |= span_hi >= trigger_above[0]
    events = np.flatnonzero(touches).tolist()

    def gap_liquidation(a: int, b: int) -> tuple[int, float] | None:
        p_liq = _liquidation_price(inv, cash, pos, mm)
        if b <= a or p_liq is None:
            return None
        hits = np.flatnonzero(span_lo[a:b] <= p_liq if pos > 0 else span_hi[a:b] >= p_liq)
        return (a + int(hits[0]), p_liq) if hits.size else None

    pos_at = np.empty(m)
    cash_at = np.empty(m)
    stopped: str | None = None
    stop_index = m
    nxt = 0

    buy_list, sell_list = buy_levels.tolist(), sell_levels.tolist()
    qty_list, short_list = qty.tolist(), short_cell.tolist()

    for k in [*events, m]:
        hit = gap_liquidation(nxt, k)
        if hit is not None:
            k, p_liq = hit
            pos_at[nxt:k] = pos
            cash_at[nxt:k] = cash
            trades.append((stamps[k], "sell" if pos > 0 else "buy", p_liq, abs(pos), -1, "liq"))
            close_position(p_liq)
            stopped, stop_index = "liquidation", k
            break
        pos_at[nxt:k] = pos
        cash_at[nxt:k] = cash
        if k == m:
            break

        if c[k] >= o[k]:
            path = (o[k], lo[k], h[k], c[k])
        else:
            path = (o[k], h[k], lo[k], c[k])
        price = float(prev_close[k])

        for target in path:
            target = float(target)
            trigger: str | None = None
            if trigger_below is not None and target <= trigger_below[0]:
                target, trigger = trigger_below
            elif trigger_above is not None and target >= trigger_above[0]:
                target, trigger = trigger_above

            if target < price:
                # Bajada: se ejecutan las compras con nivel en [target, price).
                i = int(np.searchsorted(buy_levels, target, "left"))
                while i < n and buy_list[i] < price:
                    if not holding[i]:
                        q, p = qty_list[i], buy_list[i]
                        fee = q * p * maker
                        cash -= q * p + fee
                        fees_paid += fee
                        grid_profit -= fee
                        if short_list[i]:
                            grid_profit += q * (sell_list[i] - p)
                            round_trips += 1
                        pos += q
                        holding[i] = True
                        trades.append((stamps[k], "buy", p, q, i, "grid"))
                    i += 1
            elif target > price:
                # Subida: se ejecutan las ventas con nivel en (price, target].
                i = int(np.searchsorted(sell_levels, price, "right"))
                while i < n and sell_list[i] <= target:
                    if holding[i]:
                        q, p = qty_list[i], sell_list[i]
                        fee = q * p * maker
                        cash += q * p - fee
                        fees_paid += fee
                        grid_profit -= fee
                        if not short_list[i]:
                            grid_profit += q * (p - buy_list[i])
                            round_trips += 1
                        pos -= q
                        holding[i] = False
                        trades.append((stamps[k], "sell", p, q, i, "grid"))
                    i += 1

            p_liq = _liquidation_price(inv, cash, pos, mm)
            if p_liq is not None and (
                (pos > 0 and target <= p_liq) or (pos < 0 and target >= p_liq)
            ):
                trigger, target = "liquidation", p_liq
            price = target

            if trigger is not None:
                if pos != 0:
                    side = "sell" if pos > 0 else "buy"
                    reason = "liq" if trigger == "liquidation" else trigger
                    trades.append((stamps[k], side, price, abs(pos), -1, reason))
                    close_position(price)
                holding[:] = short_cell  # sin posición en ninguna celda
                stopped, stop_index = trigger, k
                break
        pos_at[k] = pos
        cash_at[k] = cash
        nxt = k + 1
        if stopped:
            break

    if stopped:
        pos_at[stop_index:] = 0.0
        cash_at[stop_index:] = cash
        if stopped == "liquidation":
            # Margen aislado: lo que queda tras liquidar se pierde.
            cash_at[stop_index:] = -inv
    active_end = stop_index + 1 if stopped else m

    # Funding: pagos sobre la posición vigente en cada liquidación de funding.
    funding_cum = np.zeros(m)
    funding_pnl = 0.0
    if funding is not None and not funding.empty:
        f_stamps = pd.DatetimeIndex(funding["ts"])
        if f_stamps.tz is None:
            f_stamps = f_stamps.tz_localize("UTC")
        idx = np.searchsorted(stamps.asi8, f_stamps.asi8, "left")
        valid = (f_stamps.asi8 >= stamps.asi8[0]) & (idx < min(active_end, m))
        idx = idx[valid]
        if idx.size:
            payments = -pos_at[idx] * o[idx] * funding["rate"].to_numpy()[valid]
            np.add.at(funding_cum, idx, payments)
            funding_cum = np.cumsum(funding_cum)
            funding_pnl = float(payments.sum())

    equity_values = inv + cash_at + pos_at * c + funding_cum
    if stopped == "liquidation":
        equity_values[stop_index:] = 0.0

    if exit_price is not None and not stopped and pos != 0:
        side = "sell" if pos > 0 else "buy"
        trades.append((stamps[-1], side, float(exit_price), abs(pos), -1, "close"))
        close_position(float(exit_price), taker if exit_fee is None else exit_fee)
        equity_values[-1] = inv + cash + funding_cum[-1]

    closes = c[:active_end]
    in_range = (closes >= cfg.lower) & (closes <= cfg.upper)
    range_exits = int(np.count_nonzero(in_range[:-1] & ~in_range[1:]))

    peak = np.maximum.accumulate(np.maximum(equity_values, 1e-12))
    max_drawdown = float(np.max(1 - equity_values / peak))

    return GridResult(
        config=cfg,
        start_price=p0,
        end_price=float(c[-1]),
        final_equity=float(equity_values[-1]),
        final_position=float(pos_at[-1]) if exit_price is None else pos,
        grid_profit=grid_profit,
        fees_paid=fees_paid,
        funding_pnl=funding_pnl,
        round_trips=round_trips,
        buy_hold_equity=inv * (1 - taker) / p0 * float(c[-1]),
        time_in_range=float(in_range.mean()),
        range_exits=range_exits,
        max_drawdown=max_drawdown,
        minutes=len(closes),
        stopped=stopped,
        stopped_at=stamps[stop_index] if stopped else None,
        equity=pd.Series(equity_values, index=stamps, name="equity"),
        trades=pd.DataFrame(trades, columns=["ts", "side", "price", "qty", "cell", "reason"]),
    )
