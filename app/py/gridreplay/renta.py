"""Informe para la Renta (IRPF, España) a partir del export completo de Pionex.

Criterio aplicado (el habitual de las herramientas fiscales cripto; la AEAT no
tiene una guía específica para bots, así que el usuario debe revisarlo):

* Cada operación spot o *convert* es una permuta: se transmite la moneda que
  sale y se adquiere la que entra, ambas por su valor en euros en ese momento.
  USDT/USDC también son monedas virtuales: sus lotes se llevan en FIFO y su
  ganancia o pérdida es la variación EUR/USD entre compra y uso.
* FIFO por moneda (art. 37.2 LIRPF, valores homogéneos). Comisiones: en la
  compra reducen la cantidad recibida; en la venta reducen el valor de
  transmisión.
* Futuros: resultado de cada posición cerrada (pnl + comisiones) más el
  funding cobrado/pagado → ganancia o pérdida patrimonial por derivados.
* Staking (issued/claimed_profit): rendimiento del capital mobiliario al valor
  en euros al recibirlo, que pasa a ser el coste del lote.
* Depósitos externos: coste = valor en euros al depositar (aproximación; el
  coste real es el de la compra en el exchange de origen). Retiradas: salen de
  los lotes sin tributar (no es transmisión).

Todo es puro pandas: `build_report` recibe los DataFrames y una función de
tipos de cambio, para que pueda correr también en el navegador (Pyodide).
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

STABLES = {"USDT", "USDC", "FDUSD", "BUSD"}
ALIASES = {"WLFISPOT": "WLFI"}  # Pionex renombra algunos tokens en `convert`
MODELO_721_LIMIT = 50_000.0
EPS = 1e-9

# precio(coin, ts) -> USDT por unidad; eur_per_usdt(ts) -> EUR por USDT
PriceFn = Callable[[str, pd.Timestamp], float | None]
EurFn = Callable[[pd.Timestamp], float]


def norm(coin: str) -> str:
    coin = str(coin).upper()
    return ALIASES.get(coin, coin)


# ---------------------------------------------------------------- lectura


def load_full_export(path: str | Path | bytes) -> dict[str, pd.DataFrame]:
    """Lee todos los CSV del zip (ruta, bytes o carpeta descomprimida)."""
    if isinstance(path, (str, Path)) and Path(path).is_dir():
        files = {p.name: p.read_bytes() for p in Path(path).glob("*.csv")}
    else:
        src = io.BytesIO(path) if isinstance(path, bytes) else Path(path)
        with zipfile.ZipFile(src) as zf:
            files = {n: zf.read(n) for n in zf.namelist() if n.endswith(".csv")}
    out = {}
    for name, raw in files.items():
        df = pd.read_csv(io.BytesIO(raw))
        for col in ("date(UTC+0)", "close_time"):
            if col in df.columns:
                df["ts"] = pd.to_datetime(df[col], utc=True)
                break
        out[name.removesuffix(".csv")] = df
    return out


# ---------------------------------------------------------------- eventos


@dataclass
class Event:
    ts: pd.Timestamp
    kind: str  # trade | deposit | withdraw | income | futures | other
    sent: str | None = None
    sent_qty: float = 0.0
    received: str | None = None
    received_qty: float = 0.0
    usdt_value: float | None = None  # valor de mercado en USDT
    note: str = ""


def events_from_export(ex: dict[str, pd.DataFrame]) -> list[Event]:
    ev: list[Event] = []

    raw = ex.get("raw-trading-details")
    if raw is not None:
        spot = raw[(raw["market_type"] == "Spot") & (raw["state"] == "FILLED")]
        for r in spot.itertuples():
            base, quote = (norm(x) for x in r.symbol.split("_", 1))
            qty, amt, fee, fee_coin = r.executed_qty, r.amount, r.fee, norm(r.fee_coin)
            if r.side == "BUY":
                got = qty - (fee if fee_coin == base else 0.0)
                ev.append(Event(r.ts, "trade", quote, amt, base, got, amt))
            else:
                got = amt - (fee if fee_coin == quote else 0.0)
                ev.append(Event(r.ts, "trade", base, qty, quote, got, got))

    conv = ex.get("convert")
    if conv is not None:
        for r in conv.itertuples():
            buy = norm(r.buy_coin)
            val = r.buy_amount if buy in STABLES else None
            ev.append(Event(r.ts, "trade", norm(r.sell_coin), r.sell_amount, buy, r.buy_amount,
                            val, "convert"))

    dust = ex.get("dust-collector")
    if dust is not None:
        for r in dust.itertuples():
            ev.append(Event(r.ts, "trade", norm(r.coin), r.amount, "USDT", r.swap_value,
                            r.swap_value, "dust"))

    dw = ex.get("deposit-withdraw")
    if dw is not None:
        for r in dw.itertuples():
            coin = norm(r.coin)
            if r.tx_type == "DEPOSIT":
                ev.append(Event(r.ts, "deposit", received=coin, received_qty=r.amount))
            else:
                ev.append(Event(r.ts, "withdraw", sent=coin, sent_qty=r.amount + (r.fee if pd.notna(r.fee) else 0)))

    stk = ex.get("staking")
    if stk is not None:
        inc = stk[stk["tag"].isin(["issued_profit", "claimed_profit"])]
        for r in inc.itertuples():
            ev.append(Event(r.ts, "income", received=norm(r._3), received_qty=r._2,
                            note=r.tag))

    pos = ex.get("position_futures")
    if pos is not None:
        for r in pos.itertuples():
            net = r.pnl + r.fee  # funding va aparte (others.csv es el flujo real)
            ev.append(Event(r.ts, "futures", usdt_value=net, note=r.symbol))

    oth = ex.get("others")
    if oth is not None:
        for r in oth.itertuples():
            kind = "futures" if r.tag == "FundingFee" else "other"
            ev.append(Event(r.ts, kind, usdt_value=r.amount if norm(r.coin) in STABLES else None,
                            sent=None, received=norm(r.coin), received_qty=r.amount,
                            note=r.tag))

    ev.sort(key=lambda e: e.ts)
    return ev


# ---------------------------------------------------------------- FIFO


@dataclass
class Disposal:
    ts: pd.Timestamp
    coin: str
    qty: float
    proceeds_eur: float
    cost_eur: float
    unknown_cost_qty: float = 0.0

    @property
    def gain(self) -> float:
        return self.proceeds_eur - self.cost_eur


@dataclass
class Report:
    disposals: list[Disposal] = field(default_factory=list)
    futures: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)  # ts, nota, eur
    income: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)
    other: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)
    holdings_eur: dict[int, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Lots:
    def __init__(self) -> None:
        self.q: dict[str, deque[list[float]]] = defaultdict(deque)  # [qty, eur por unidad]

    def add(self, coin: str, qty: float, eur_total: float) -> None:
        if qty > EPS:
            self.q[coin].append([qty, eur_total / qty])

    def take(self, coin: str, qty: float) -> tuple[float, float]:
        """Saca `qty` en FIFO. Devuelve (coste en euros, cantidad sin lote)."""
        cost, left, lots = 0.0, qty, self.q[coin]
        while left > EPS and lots:
            lot = lots[0]
            use = min(left, lot[0])
            cost += use * lot[1]
            lot[0] -= use
            left -= use
            if lot[0] <= EPS:
                lots.popleft()
        return cost, max(left, 0.0)

    def balances(self) -> dict[str, float]:
        return {c: sum(l[0] for l in lots) for c, lots in self.q.items() if lots}


def build_report(events: list[Event], price: PriceFn, eur: EurFn,
                 year_ends: list[int] = ()) -> Report:
    rep, lots = Report(), Lots()
    pending = sorted(year_ends)
    missing: set[str] = set()

    def usdt_value(coin: str, qty: float, ts: pd.Timestamp) -> float | None:
        if coin in STABLES:
            return qty
        p = price(coin, ts)
        if p is None:
            missing.add(coin)
        return None if p is None else qty * p

    def snapshot(year: int) -> None:
        ts = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")
        vals = {}
        for coin, qty in lots.balances().items():
            v = usdt_value(coin, qty, ts)
            if v is not None and v * eur(ts) > 1:
                vals[coin] = v * eur(ts)
        rep.holdings_eur[year] = vals

    def dispose(ts, coin, qty, proceeds_eur):
        cost, unknown = lots.take(coin, qty)
        rep.disposals.append(Disposal(ts, coin, qty, proceeds_eur, cost, unknown))

    for e in events:
        while pending and e.ts.year > pending[0]:
            snapshot(pending.pop(0))
        rate = eur(e.ts)
        if e.kind == "trade":
            val = e.usdt_value
            if val is None:
                val = usdt_value(e.sent, e.sent_qty, e.ts)
            if val is None:
                rep.warnings.append(f"{e.ts:%Y-%m-%d} sin precio para {e.sent}/{e.received}")
                continue
            dispose(e.ts, e.sent, e.sent_qty, val * rate)
            lots.add(e.received, e.received_qty, val * rate)
        elif e.kind == "deposit":
            val = usdt_value(e.received, e.received_qty, e.ts)
            if val is None:
                val = 0.0
                rep.warnings.append(f"{e.ts:%Y-%m-%d} depósito de {e.received} sin precio: coste 0")
            elif e.received not in STABLES:
                rep.warnings.append(
                    f"{e.ts:%Y-%m-%d} depósito de {e.received_qty:g} {e.received}: coste = valor "
                    f"al depositar ({val * rate:,.2f} €). Si lo compraste antes, usa tu coste real.")
            lots.add(e.received, e.received_qty, val * rate)
        elif e.kind == "withdraw":
            lots.take(e.sent, e.sent_qty)
        elif e.kind == "income":
            val = usdt_value(e.received, e.received_qty, e.ts) or 0.0
            rep.income.append((e.ts, f"{e.note} {e.received}", val * rate))
            lots.add(e.received, e.received_qty, val * rate)
        elif e.kind in ("futures", "other"):
            val = e.usdt_value or 0.0
            (rep.futures if e.kind == "futures" else rep.other).append((e.ts, e.note, val * rate))
            # el resultado se cobra/paga en USDT: entra como lote o consume lotes
            if val > 0:
                lots.add("USDT", val, val * rate)
            elif val < 0:
                dispose(e.ts, "USDT", -val, -val * rate)
    for y in pending:
        snapshot(y)
    for coin in sorted(missing):
        rep.warnings.append(f"no encontré precio en Binance para {coin}")
    return rep


# ---------------------------------------------------------------- resumen


def summarize(rep: Report, year: int) -> dict:
    d = [x for x in rep.disposals if x.ts.year == year and x.qty > EPS]
    by_coin: dict[str, dict] = {}
    for x in d:
        c = by_coin.setdefault(x.coin, {"operaciones": 0, "transmision": 0.0, "adquisicion": 0.0})
        c["operaciones"] += 1
        c["transmision"] += x.proceeds_eur
        c["adquisicion"] += x.cost_eur
    for c in by_coin.values():
        c["resultado"] = c["transmision"] - c["adquisicion"]
    fut = [v for ts, _, v in rep.futures if ts.year == year]
    inc = sum(v for ts, _, v in rep.income if ts.year == year)
    oth: dict[str, float] = defaultdict(float)
    for ts, note, v in rep.other:
        if ts.year == year:
            oth[note] += v
    unknown: dict[str, float] = defaultdict(float)
    for x in d:
        if x.unknown_cost_qty > 1e-6:
            unknown[x.coin] += x.unknown_cost_qty
    holdings = rep.holdings_eur.get(year, {})
    total_h = sum(holdings.values())
    crypto = sum(c["resultado"] for c in by_coin.values())
    return {
        "ejercicio": year,
        "monedas_virtuales": {
            "por_moneda": dict(sorted(by_coin.items(), key=lambda kv: -abs(kv[1]["resultado"]))),
            "ganancias": sum(x.gain for x in d if x.gain > 0),
            "perdidas": sum(x.gain for x in d if x.gain < 0),
            "neto": crypto,
            "transmisiones": len(d),
        },
        "derivados_futuros": {"neto": sum(fut), "movimientos": len(fut)},
        "capital_mobiliario_staking": inc,
        "otros_a_revisar": dict(oth),
        "base_ahorro_estimada": crypto + sum(fut) + inc + sum(oth.values()),
        "saldo_31_dic": {"total_eur": total_h, "por_moneda": holdings,
                         "modelo_721": total_h > MODELO_721_LIMIT},
        "ventas_sin_coste_conocido": dict(unknown),
    }


def format_summary(s: dict, warnings: list[str] = ()) -> str:
    e = lambda v: f"{v:>12,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")  # noqa: E731
    mv, fut = s["monedas_virtuales"], s["derivados_futuros"]
    out = [f"INFORME RENTA {s['ejercicio']} · Pionex (borrador, revísalo con tu asesor)", "",
           f"1. Monedas virtuales (permutas y ventas, FIFO) · {mv['transmisiones']} transmisiones",
           f"   {'Moneda':<8}{'Transmisión':>15}{'Adquisición':>15}{'Resultado':>15}"]
    for coin, c in mv["por_moneda"].items():
        out.append(f"   {coin:<8}{e(c['transmision'])}{e(c['adquisicion'])}{e(c['resultado'])}")
    out += [f"   Ganancias {e(mv['ganancias'])}   Pérdidas {e(mv['perdidas'])}   "
            f"Neto {e(mv['neto'])}", "",
            f"2. Derivados (futuros + funding) · {fut['movimientos']} movimientos",
            f"   Neto {e(fut['neto'])}", "",
            f"3. Capital mobiliario (staking)   {e(s['capital_mobiliario_staking'])}"]
    if s["otros_a_revisar"]:
        out += ["", "4. Otros a revisar (bonos, compensaciones)"]
        out += [f"   {k:<14}{e(v)}" for k, v in s["otros_a_revisar"].items()]
    h = s["saldo_31_dic"]
    out += ["", f"Base del ahorro estimada        {e(s['base_ahorro_estimada'])}", "",
            f"Saldo en Pionex a 31/12: {e(h['total_eur']).strip()} → Modelo 721: "
            + ("SÍ, supera 50.000 €" if h["modelo_721"] else "no (≤ 50.000 €)")]
    if s["ventas_sin_coste_conocido"]:
        out += ["", "Ventas sin lote de compra (coste 0, revisa compras fuera de Pionex):"]
        out += [f"   {k}: {v:g}" for k, v in s["ventas_sin_coste_conocido"].items()]
    if warnings:
        out += ["", "Avisos:"] + [f"   · {w}" for w in warnings]
    return "\n".join(out)


def disposals_csv(rep: Report, year: int) -> str:
    rows = [{"fecha_utc": x.ts.strftime("%Y-%m-%d %H:%M:%S"), "moneda": x.coin,
             "cantidad": round(x.qty, 10), "valor_transmision_eur": round(x.proceeds_eur, 4),
             "valor_adquisicion_eur": round(x.cost_eur, 4), "resultado_eur": round(x.gain, 4)}
            for x in rep.disposals if x.ts.year == year and x.qty > EPS]
    return pd.DataFrame(rows).to_csv(index=False)


# ---------------------------------------------------------------- precios (Binance)

KLINES = "https://api.binance.com/api/v3/klines"
HOUR_MS = 3_600_000
PAGE = 1000  # velas por petición


def needed_symbols(events: list[Event]) -> list[str]:
    coins = {c for e in events for c in (e.sent, e.received) if c and c not in STABLES}
    return ["EURUSDT", *sorted(f"{c}USDT" for c in coins)]


def price_range(events: list[Event]) -> tuple[int, int]:
    """Milisegundos [inicio, fin] de las velas necesarias (hasta el 31/12 del último año o hoy)."""
    start = events[0].ts.floor("h")
    end = min(pd.Timestamp(f"{events[-1].ts.year}-12-31 23:00", tz="UTC"),
              pd.Timestamp.now(tz="UTC").floor("h"))
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def page_urls(symbol: str, start: int, end: int) -> list[str]:
    """Páginas independientes (se pueden pedir en paralelo)."""
    return [f"{KLINES}?symbol={symbol}&interval=1h&startTime={t}"
            f"&endTime={min(t + PAGE * HOUR_MS - 1, end)}&limit={PAGE}"
            for t in range(start, end + 1, PAGE * HOUR_MS)]


class PriceTable:
    """Cierres horarios por símbolo: {símbolo: [[ms, cierre], ...]} (vacío = el par no existe)."""

    def __init__(self, data: dict[str, list]) -> None:
        self.series: dict[str, pd.Series | None] = {}
        for sym, rows in data.items():
            if rows:
                s = pd.Series([float(r[1]) for r in rows],
                              index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
                self.series[sym] = s[~s.index.duplicated()].sort_index()

    def _at(self, symbol: str, ts: pd.Timestamp) -> float | None:
        s = self.series.get(symbol)
        if s is None:
            return None
        i = s.index.searchsorted(ts, side="right") - 1
        return float(s.iloc[max(i, 0)])

    def price(self, coin: str, ts: pd.Timestamp) -> float | None:
        return self._at(f"{coin}USDT", ts)

    def eur(self, ts: pd.Timestamp) -> float:
        rate = self._at("EURUSDT", ts)
        if rate is None:
            raise OSError("No se pudo descargar el cambio EUR/USDT de Binance")
        return 1.0 / rate


def _closes(body: bytes | list) -> list:
    rows = json.loads(body) if isinstance(body, (bytes, str)) else body
    return [[k[0], float(k[4])] for k in rows]


def fetch_prices(events: list[Event], cache: Path = Path("data/renta_cache")) -> PriceTable:
    """Versión de escritorio: requests + caché en disco."""
    import requests

    start, end = price_range(events)
    cache.mkdir(parents=True, exist_ok=True)
    data = {}
    for sym in needed_symbols(events):
        f = cache / f"{sym}_{start}_{end}.json"
        if not f.exists():
            rows: list = []
            for url in page_urls(sym, start, end):
                r = requests.get(url, timeout=20)
                if r.status_code == 400:  # el par no existe en Binance
                    rows = []
                    break
                r.raise_for_status()
                rows += _closes(r.content)
                time.sleep(0.05)
            f.write_text(json.dumps(rows))
        data[sym] = json.loads(f.read_text())
    return PriceTable(data)


def _year_results(rep: Report, years: list[int]) -> dict:
    return {y: {"summary": summarize(rep, y), "csv": disposals_csv(rep, y)} for y in years}


def run(path: str | Path, year: int) -> tuple[Report, dict, list[Event]]:
    events = events_from_export(load_full_export(path))
    px = fetch_prices(events)
    years = sorted({e.ts.year for e in events} | {year})
    rep = build_report(events, px.price, px.eur, years)
    return rep, summarize(rep, year), events


async def report_web(export: bytes, get, progress=None) -> dict:
    """Versión navegador (Pyodide): `get(url)` asíncrono → (estado, cuerpo)."""
    import asyncio

    events = events_from_export(load_full_export(export))
    if not events:
        return {"error": "empty"}
    start, end = price_range(events)
    symbols = needed_symbols(events)
    done, total = 0, len(symbols)

    async def one(sym: str) -> tuple[str, list]:
        nonlocal done
        rows: list = []
        for status, body in await asyncio.gather(*(get(u) for u in page_urls(sym, start, end))):
            if status == 400:
                rows = []
                break
            if status != 200:
                raise OSError(f"HTTP {status} al descargar {sym}")
            rows += _closes(body)
        done += 1
        if progress:
            progress({"done": done, "total": total, "symbol": sym})
        return sym, rows

    px = PriceTable(dict(await asyncio.gather(*(one(s) for s in symbols))))
    years = sorted({e.ts.year for e in events})
    rep = build_report(events, px.price, px.eur, years)
    now = pd.Timestamp.now(tz="UTC")
    return {"years": {str(y): {**r, "partial": y >= now.year}
                      for y, r in _year_results(rep, years).items()},
            "warnings": rep.warnings, "events": len(events)}
