"""Comisiones de grid por exchange (tarifa base, sin descuentos por token).

Revisar periódicamente: los exchanges cambian sus tarifas. Las de Pionex
futuros están verificadas contra exports reales (0,02 % maker, 0,05 % taker).
"""

SPOT_FEES: dict[str, float] = {
    "pionex": 0.0005,
    "binance": 0.001,
    "bybit": 0.001,
    "kucoin": 0.001,
}

# (maker, taker) en perpetuos USDT.
FUTURES_FEES: dict[str, tuple[float, float]] = {
    "pionex": (0.0002, 0.0005),
    "binance": (0.0002, 0.0005),
    "bybit": (0.0002, 0.00055),
    "kucoin": (0.0002, 0.0006),
}


def fee_for(exchange: str) -> float:
    return fees_for(exchange, futures=False)[0]


def fees_for(exchange: str, futures: bool) -> tuple[float, float]:
    """(maker, taker) del exchange; en spot ambas son iguales."""
    table = FUTURES_FEES if futures else {k: (v, v) for k, v in SPOT_FEES.items()}
    try:
        return table[exchange.lower()]
    except KeyError:
        known = ", ".join(sorted(table))
        raise ValueError(f"Exchange desconocido '{exchange}'. Disponibles: {known}") from None
