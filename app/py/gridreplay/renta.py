"""Informe para la Renta (IRPF, España) a partir del historial de OKX (beta) o de Pionex.

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
import re
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
    return _frames(files)


def _frames(files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
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


# ---------------------------------------------------------------- OKX (beta)
#
# OKX no publica el formato de sus descargas. Las columnas son las que usa BittyTax
# (src/bittytax/conv/parsers/okx.py), comprobadas en noviembre de 2026; pendiente de
# confirmar con un export real (sobre todo: zona horaria y si salen los grid bots).
#   Historial de trading: id, Order id, Time, Trade Type, Instrument, Type, Amount, Unit,
#     PL, Fee, Position Change, Position Balance, Balance Change, Balance, Unit
#     → cada operación spot son dos filas (Buy: la moneda que entra; Sell: la que sale)
#       con el mismo id (o, si no, el mismo Order id).
#   Historial de fondos: id, [columna vacía], Time, Type, Amount, Before Balance,
#     After Balance, Fee, Symbol
# Ante cualquier tipo de movimiento desconocido no se calcula nada (OKXUnsupported):
# en una herramienta fiscal es mejor no dar cifra que dar una mala.

OKX_TRADING = ["Order id", "Time", "Trade Type", "Instrument", "Type", "Amount", "Unit",
               "Fee", "Balance Change"]
OKX_FUNDING = ["Time", "Type", "Amount", "Before Balance", "After Balance", "Fee", "Symbol"]
OKX_FUNDING_SKIP = {"Stake", "Redeem staking", "Savings subscription", "Savings redemption",
                    "From unified trading account", "To unified trading account"}
OKX_TZ = {"UTC": "UTC", "UTC+8": "Asia/Shanghai", "Madrid": "Europe/Madrid"}


class OKXUnsupported(ValueError):
    """El export de OKX trae algo que todavía no sabemos tratar."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def read_files(files: dict[str, bytes]) -> dict[str, bytes]:
    """Abre los .zip y devuelve {nombre: bytes} de todos los CSV."""
    out: dict[str, bytes] = {}
    for name, raw in files.items():
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".csv"):
                        out[Path(n).name] = zf.read(n)
        elif name.lower().endswith(".csv"):
            out[Path(name).name] = raw
    return out


def _okx_frame(raw: bytes) -> tuple[str, pd.DataFrame] | None:
    """('trading' | 'funding', tabla) si el CSV es de OKX; None si no."""
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines[:10]):  # puede haber filas de cuenta/zona horaria antes
        cells = [c.strip().strip('"').lstrip("﻿") for c in line.split(",")]
        if not cells or cells[0] != "id":
            continue
        df = pd.read_csv(io.StringIO("\n".join(lines[i:])), dtype=str)
        df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
        if all(c in df.columns for c in OKX_TRADING):
            return "trading", df
        if all(c in df.columns for c in OKX_FUNDING):
            return "funding", df
    return None


PIONEX_FILES = {"raw-trading-details", "convert", "dust-collector", "deposit-withdraw", "staking",
                "position_futures", "others"}


def detect(files: dict[str, bytes]) -> str:
    """'okx', 'pionex' o 'unknown' (otro exchange o archivo que no reconocemos)."""
    if any(_okx_frame(raw) for raw in files.values()):
        return "okx"
    if any(Path(n).stem in PIONEX_FILES for n in files):
        return "pionex"
    return "unknown"


def load_okx(files: dict[str, bytes], tz: str = "UTC") -> dict[str, pd.DataFrame]:
    zone = OKX_TZ.get(tz, tz)
    parts: dict[str, list[pd.DataFrame]] = {"trading": [], "funding": []}
    for raw in files.values():
        got = _okx_frame(raw)
        if got:
            parts[got[0]].append(got[1])
    out = {}
    for kind, frames in parts.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates()
        ts = pd.to_datetime(df["Time"].str.strip(), errors="coerce")
        if ts.isna().any():
            raise OKXUnsupported("okx_time", str(df.loc[ts.isna(), "Time"].iloc[0]))
        df["ts"] = ts.dt.tz_localize(zone).dt.tz_convert("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        out[kind] = df
    return out


def _num(v) -> float:
    s = str(v).strip().replace(",", "")
    return 0.0 if s in ("", "nan", "--") else float(s)


def events_from_okx(ok: dict[str, pd.DataFrame]) -> list[Event]:
    ev: list[Event] = []
    tr = ok.get("trading")
    if tr is not None:
        unknown = sorted(set(tr["Trade Type"].str.strip()) - {"Spot", "Transfer"})
        if unknown:
            raise OKXUnsupported("okx_types", ", ".join(unknown))
        spot = tr[tr["Trade Type"].str.strip() == "Spot"].copy()
        bad = sorted(set(spot["Type"].str.strip()) - {"Buy", "Sell"})
        if bad:
            raise OKXUnsupported("okx_types", ", ".join(bad))
        # El segundo «Unit» (moneda del saldo) lo renombra pandas a «Unit.1»: se usa el primero.
        spot["key"] = spot["id"].where(spot.groupby("id")["id"].transform("size") == 2,
                                       "order:" + spot["Order id"])
        for _, g in spot.groupby("key", sort=False):
            buy, sell = g[g["Type"].str.strip() == "Buy"], g[g["Type"].str.strip() == "Sell"]
            if len(buy) != 1 or len(sell) != 1:
                raise OKXUnsupported("okx_pairs", str(g["Instrument"].iloc[0]))
            b, s = buy.iloc[0], sell.iloc[0]
            got = _num(b["Balance Change"]) or _num(b["Amount"]) - abs(_num(b["Fee"]))
            gave = abs(_num(s["Balance Change"])) or abs(_num(s["Amount"]))
            if got <= 0 or gave <= 0:
                raise OKXUnsupported("okx_amounts", str(b["Instrument"]))
            rcv, snt = norm(b["Unit"]), norm(s["Unit"])
            val = gave if snt in STABLES else got if rcv in STABLES else None
            ev.append(Event(min(b["ts"], s["ts"]), "trade", snt, gave, rcv, got, val))

    fu = ok.get("funding")
    if fu is not None:
        types = fu["Type"].str.strip()
        known = {"Deposit", "Withdrawal", "Staking Yield", "Fee rebate"} | OKX_FUNDING_SKIP
        unknown = sorted(set(types) - known)
        if unknown:
            raise OKXUnsupported("okx_types", ", ".join(unknown))
        for (_, r), t in zip(fu.iterrows(), types, strict=True):
            coin, amt, fee = norm(r["Symbol"]), abs(_num(r["Amount"])), abs(_num(r["Fee"]))
            if t == "Deposit":
                ev.append(Event(r["ts"], "deposit", received=coin, received_qty=amt))
            elif t == "Withdrawal":
                ev.append(Event(r["ts"], "withdraw", sent=coin, sent_qty=amt + fee))
            elif t == "Staking Yield":
                ev.append(Event(r["ts"], "income", received=coin, received_qty=amt, note="staking"))
            elif t == "Fee rebate":
                ev.append(Event(r["ts"], "other", received=coin, received_qty=amt,
                                usdt_value=amt if coin in STABLES else None,
                                note="Reembolso de comisiones"))
    if tr is None:
        raise OKXUnsupported("okx_no_trading")
    ev.sort(key=lambda e: e.ts)
    return ev


# ---------------------------------------------------------------- estructura (opcional)
#
# Para aprender formatos que no conocemos (OKX, otros exchanges), el usuario puede enviar
# voluntariamente SOLO la estructura de su archivo: nombres de columnas, valores de las
# columnas de tipo (sin cifras) y la forma de la fecha con los dígitos cambiados por 9.
# Nunca importes, fechas reales, monedas ni identificadores.

_TYPE_COL = re.compile(r"type|side|action|tag|categor|kind|status|state|direction|tipo|bill", re.I)
_TIME_COL = re.compile(r"time|date|fecha", re.I)
_SAFE_VALUE = re.compile(r"^[A-Za-z][A-Za-z /_&().'-]{0,39}$")  # sin dígitos ni @


def _mask(text: str, limit: int = 120) -> str:
    text = re.sub(r"[^\s,;]*@[^\s,;]*", "[email]", str(text))  # sin @: el servidor las rechaza
    return re.sub(r"\d", "9", text)[:limit]


def _size_bucket(n: int) -> str:
    return "0" if n == 0 else "<10" if n < 10 else "<100" if n < 100 else "<1000" if n < 1000 else "1000+"


def file_structure(files: dict[str, bytes]) -> dict:
    csvs = read_files(files)
    out = {"v": 1, "exchange": detect(csvs) if csvs else "unknown",
           "others": sorted({_mask(Path(n).suffix.lower(), 10) for n in files
                             if not n.lower().endswith((".csv", ".zip"))})[:10],
           "files": []}
    for name, raw in list(csvs.items())[:12]:
        lines = raw.decode("utf-8-sig", errors="replace").splitlines()
        head = next((i for i, l in enumerate(lines[:15])
                     if l.count(",") >= 2 and sum(c.isalpha() for c in l) > sum(c.isdigit() for c in l)), 0)
        info = {"name": _mask(Path(name).name, 80), "before_header": [_mask(l) for l in lines[:head][:5]]}
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype=str, nrows=200_000)
        except Exception:  # noqa: BLE001 — CSV ilegible: se manda solo lo anterior
            out["files"].append({**info, "error": "unreadable"})
            continue
        cols = [str(c).strip().lstrip("﻿") for c in df.columns][:80]
        df.columns = cols + list(df.columns[len(cols):])
        info["columns"] = [_mask(c, 60) for c in cols]
        info["rows"] = _size_bucket(len(df))
        info["values"] = {}
        for c in cols:
            if _TYPE_COL.search(c):
                vals = df[c].dropna().astype(str).str.strip().unique()
                safe = sorted(v for v in vals if _SAFE_VALUE.match(v))[:30]
                if safe:
                    info["values"][_mask(c, 60)] = safe
        tcol = next((c for c in cols if _TIME_COL.search(c)), None)
        if tcol is not None and df[tcol].notna().any():
            info["time_shape"] = _mask(df[tcol].dropna().iloc[0], 40)
        out["files"].append(info)
    return out


def load_events(files: dict[str, bytes], tz: str = "UTC") -> tuple[str, list[Event]]:
    """(exchange, eventos) a partir de lo que suba el usuario (zips o CSV, OKX o Pionex)."""
    csvs = read_files(files)
    if detect(csvs) == "okx":
        return "OKX", events_from_okx(load_okx(csvs, tz))
    return "Pionex", events_from_export(_frames(csvs))


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


def format_summary(s: dict, warnings: list[str] = (), exchange: str = "Pionex") -> str:
    exchange = s.get("exchange", exchange)
    e = lambda v: f"{v:>12,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")  # noqa: E731
    mv, fut = s["monedas_virtuales"], s["derivados_futuros"]
    out = [f"INFORME RENTA {s['ejercicio']} · {exchange} (borrador, revísalo con tu asesor)", "",
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
            f"Saldo en {exchange} a 31/12: {e(h['total_eur']).strip()} → Modelo 721: "
            + ("SÍ, supera 50.000 €" if h["modelo_721"] else "no (≤ 50.000 €)")]
    if s["ventas_sin_coste_conocido"]:
        out += ["", f"Ventas sin lote de compra (coste 0, revisa compras fuera de {exchange}):"]
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


def run(path: str | Path, year: int, tz: str = "UTC") -> tuple[Report, dict, list[Event]]:
    """Escritorio: `path` es un zip o CSV (Pionex u OKX) o una carpeta con varios."""
    p = Path(path)
    files = {f.name: f.read_bytes() for f in p.iterdir() if f.is_file()} if p.is_dir() else {p.name: p.read_bytes()}
    exchange, events = load_events(files, tz)
    px = fetch_prices(events)
    years = sorted({e.ts.year for e in events} | {year})
    rep = build_report(events, px.price, px.eur, years)
    return rep, {**summarize(rep, year), "exchange": exchange}, events


async def report_web(export: bytes | dict[str, bytes], get, progress=None, tz: str = "UTC") -> dict:
    """Versión navegador (Pyodide): `get(url)` asíncrono → (estado, cuerpo).

    `export`: el zip de Pionex (bytes) o {nombre: bytes} con lo que suba el usuario."""
    import asyncio

    files = export if isinstance(export, dict) else {"export.zip": export}
    try:
        exchange, events = load_events(files, tz)
    except OKXUnsupported as e:
        return {"error": e.code, "detail": e.detail}
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
            "warnings": rep.warnings, "events": len(events), "exchange": exchange}
