"""Validación del simulador contra bots reales reconstruidos desde Pionex.

Para cada bot se simula el mismo grid, en el mismo periodo, sobre las velas de
1 minuto del perpetuo de Binance, y se compara con lo que ocurrió de verdad:

* ejecuciones por nivel y lado (coincidencia = Σ mín(real, sim) / máx(totales));
* PnL bruto, comisiones, funding y resultado neto.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridreplay.data import DEFAULT_CACHE, load_candles, load_funding
from gridreplay.pionex import PionexBot
from gridreplay.sim import GridConfig, GridResult, run_grid


@dataclass
class Validation:
    bot: PionexBot
    result: GridResult
    real_fills: int
    sim_fills: int
    fill_match: float

    @property
    def sim_gross(self) -> float:
        return self.result.total_pnl + self.result.fees_paid - self.result.funding_pnl

    @property
    def sim_net(self) -> float:
        return self.result.total_pnl

    @property
    def net_diff(self) -> float:
        return self.sim_net - self.bot.real_net


def bot_config(bot: PionexBot) -> GridConfig:
    """Config equivalente. La inversión real no está en el export: se usa el
    nocional a 1x, que basta para comparar importes en USDT (no porcentajes)."""
    notional = bot.qty * float(np.sum(bot.level(np.arange(bot.k_lower, bot.k_upper))))
    return GridConfig(
        lower=bot.lower,
        upper=bot.upper,
        grids=bot.grids,
        investment=notional,
        mode=bot.mode,
        fee_rate=bot.maker_fee,
        taker_fee=bot.taker_fee,
        direction=bot.direction,
        qty=bot.qty,
        entry_price=bot.entry_price,
    )


def _fill_counts(sides: pd.Series, levels: pd.Series) -> Counter:
    return Counter(zip(sides, levels, strict=True))


def sim_window(bot: PionexBot) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Periodo a simular. El bot arranca a mitad de minuto: se simula desde el
    minuto siguiente (con el estado inicial al precio de entrada real) hasta el
    minuto del cierre."""
    return (
        bot.start.floor("min") + pd.Timedelta(minutes=1),
        bot.end.floor("min") + pd.Timedelta(minutes=1),
    )


def validate_bot(bot: PionexBot, cache: Path = DEFAULT_CACHE) -> Validation:
    start, end = sim_window(bot)
    candles = load_candles(bot.binance_symbol, start, end, cache, market="futures")
    funding = load_funding(bot.binance_symbol, start, end, cache)
    return compare_bot(bot, candles, funding)


def compare_bot(bot: PionexBot, candles: pd.DataFrame, funding: pd.DataFrame) -> Validation:
    """Simula el bot sobre `candles` y `funding` ya cargados y lo compara con el real."""
    result = run_grid(
        candles,
        bot_config(bot),
        funding=funding,
        exit_price=bot.exit_price,
        exit_fee=bot.exit_fee,
    )

    sim = result.trades[result.trades["reason"] == "grid"]
    sim_levels = np.where(sim["side"] == "buy", sim["cell"], sim["cell"] + 1)
    real = _fill_counts(bot.fills["side"], bot.fills["level"])
    simulated = _fill_counts(sim["side"], pd.Series(sim_levels))
    matched = sum((real & simulated).values())
    total = max(sum(real.values()), sum(simulated.values()), 1)

    return Validation(
        bot=bot,
        result=result,
        real_fills=len(bot.fills),
        sim_fills=len(sim),
        fill_match=matched / total,
    )
