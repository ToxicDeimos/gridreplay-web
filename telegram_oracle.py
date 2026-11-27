"""Publica el oráculo de grids en el canal de Telegram de GridReplay.

Mismo cálculo que la página del oráculo (site/oracle_template.html): señal de
posicionamiento de Binance futuros + rango ±2σ de 30 días (perfil «equilibrado»).

- Resumen diario de BTC y ETH: la primera ejecución del día a partir de las 8:00.
- Aviso cuando BTC o ETH cambian de dirección y el cambio se mantiene en dos
  comprobaciones seguidas (para no avisar por ruido de 5 minutos).

Funciona dentro del proyecto (lee site/oracle.json y el .env de la raíz) o suelto, en
una carpeta cualquiera (p. ej. la Raspberry): entonces descarga los datos del oráculo de
https://gridreplay.com/oracle.json y lee el .env y guarda su estado junto al script.
Solo usa la biblioteca estándar de Python.

Configuración en `.env` (no se sube a git):
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT=@gridreplay

Uso:
    python scripts/telegram_oracle.py            # lo que toque (resumen y/o avisos)
    python scripts/telegram_oracle.py --dry      # muestra el mensaje sin enviarlo
    python scripts/telegram_oracle.py --daily    # fuerza el resumen diario ahora
    python3 telegram_oracle.py --init            # al instalarlo en otra máquina: no envía nada
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_REPO = (HERE.parent / "site" / "oracle.json").exists()
ROOT = HERE.parent if IN_REPO else HERE
ORACLE = ROOT / "site" / "oracle.json"
ORACLE_URL = "https://gridreplay.com/oracle.json"
STATE = ROOT / "data" / "telegram_state.json" if IN_REPO else HERE / "telegram_state.json"
F = "https://fapi.binance.com"
PROFILE = "equilibrado"
LEVERAGE = 2
DAILY_HOUR = 8
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
EMOJI = {"long": "🟢", "short": "🔴", "neutral": "⚪"}


def get(path: str):
    req = urllib.request.Request(F + path, headers={"User-Agent": "gridreplay-oracle"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def pct(x: float, q: list[float]) -> float:
    """Percentil de x en una lista de cuantiles (interpolado), como en la página."""
    if x <= q[0]:
        return 0.0
    if x >= q[-1]:
        return 1.0
    i = 0
    while q[i + 1] < x:
        i += 1
    span = q[i + 1] - q[i]
    return (i + ((x - q[i]) / span if span > 0 else 0)) / (len(q) - 1)


def oracle(base: str, ref: dict, k: float) -> dict:
    s = base + "USDT"
    pos = get(f"/futures/data/topLongShortPositionRatio?symbol={s}&period=5m&limit=1")
    acc = get(f"/futures/data/topLongShortAccountRatio?symbol={s}&period=5m&limit=1")
    kl = get(f"/fapi/v1/klines?symbol={s}&interval=1h&limit=338")
    t1 = get(f"/futures/data/takerlongshortRatio?symbol={s}&period=5m&limit=500")
    t2 = get(
        f"/futures/data/takerlongshortRatio?symbol={s}&period=5m&limit=364"
        f"&endTime={t1[0]['timestamp'] - 1}"
    )
    taker = [float(x["buySellRatio"]) for x in t2 + t1]
    mean = lambda a: sum(a) / len(a)  # noqa: E731
    closes = [float(c[4]) for c in kl[:-1]]  # sin la vela en curso
    rets = [math.log(c / closes[i]) for i, c in enumerate(closes[1:])][-336:]
    mu = mean(rets)
    vol = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(24)
    last = float(kl[-1][4])
    sig = (
        (1 - pct(float(pos[0]["longShortRatio"]), ref["ls_top"]))
        + (1 - pct(float(acc[0]["longShortRatio"]), ref["ls_top_acc"]))
        + pct(mean(taker), ref["taker_3d"])
        + pct(mean(taker[-288:]), ref["taker_1d"])
    ) / 4
    direction = "long" if sig >= 2 / 3 else "short" if sig <= 1 / 3 else "neutral"
    half = min(max(k * vol * math.sqrt(30), 0.02), 0.7)
    grids = math.floor(min(max(2 * half / (0.5 * vol), 8), 200) + 0.5)
    return {
        "base": base,
        "price": last,
        "dir": direction,
        "sig": sig,
        "half": half,
        "grids": grids,
        "lo": last * (1 - half),
        "hi": last * (1 + half),
        "low": pct(vol, ref["vol"]) < 0.2,
    }


STOP_BUF = 0.001  # 0,1 % más allá del límite del rango


def fmt(v: float) -> str:
    """5 cifras significativas, miles con espacio (se lee igual en ES y EN)."""
    d = max(0, 5 - len(str(int(abs(v)))) if abs(v) >= 1 else 5 - int(math.floor(math.log10(abs(v)))) - 1)
    s = f"{v:,.{d}f}".replace(",", " ")
    return s.rstrip("0").rstrip(".") if "." in s else s


def stops_line(o: dict) -> str:
    """SL y TP justo fuera del rango (scripts/oracle_stops.py, variante b0):
    el SL en el lado que pierde, el TP en el otro."""
    lo, hi = fmt(o["lo"] * (1 - STOP_BUF)), fmt(o["hi"] * (1 + STOP_BUF))
    if o["dir"] == "short":
        return f"🛑 SL {hi} · 🎯 TP {lo}"
    return f"🛑 SL {lo} · 🎯 TP {hi}"


def block(o: dict) -> str:
    lines = [
        f"<b>{o['base']}/USDT</b> · {fmt(o['price'])}",
        f"{EMOJI[o['dir']]} <b>{o['dir'].upper()}</b> · señal/signal {round(o['sig'] * 100)}",
        f"Rango/Range <b>{fmt(o['lo'])} – {fmt(o['hi'])}</b> (±{o['half'] * 100:.1f}%)",
        f"{o['grids']} grids · {LEVERAGE}x",
        stops_line(o),
    ]
    if o["low"]:
        lines.append("⚠️ Volatilidad muy baja / very low volatility")
    return "\n".join(lines)


CHANNEL_PAIRS = ["BTC", "ETH"]
FOOTER = (
    "📊 23 pares gratis en la web / 23 free pairs on the site (SOL, XRP, DOGE…)\n"
    '<a href="https://gridreplay.com/oraculo/">Oráculo</a> · '
    '<a href="https://gridreplay.com/en/oracle/">Oracle</a>\n\n'
    "<i>No es consejo financiero · Not financial advice</i>"
)


def daily_msg(results: list[dict], n_locked: int, now: datetime) -> str:
    head = f"📊 <b>Oráculo de grids / Grid oracle · {now.day} {MONTHS[now.month - 1]}</b>"
    body = "\n\n".join(block(o) for o in results)
    return f"{head}\n\n{body}\n\n{FOOTER.format(n=n_locked)}"


def change_msg(o: dict, old: str, n_locked: int) -> str:
    head = (
        f"🔄 <b>{o['base']}/USDT: {old.upper()} → {o['dir'].upper()}</b>\n"
        "Cambio de dirección confirmado / Direction change confirmed"
    )
    return f"{head}\n\n{block(o)}\n\n{FOOTER.format(n=n_locked)}"


def send(text: str, silent: bool) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    if not token or not chat:
        sys.exit("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT en .env")
    data = urllib.parse.urlencode(
        {
            "chat_id": chat,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
            "disable_notification": "true" if silent else "false",
        }
    ).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=20) as r:
        if not json.load(r).get("ok"):
            sys.exit("Telegram rechazó el mensaje")


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="mostrar sin enviar ni guardar estado")
    ap.add_argument("--daily", action="store_true", help="forzar el resumen diario")
    ap.add_argument("--init", action="store_true",
                    help="guardar el estado actual sin enviar nada (al instalarlo en otra máquina)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()

    if IN_REPO:
        o = json.loads(ORACLE.read_text(encoding="utf-8"))
    else:
        req = urllib.request.Request(ORACLE_URL, headers={"User-Agent": "gridreplay-telegram"})
        with urllib.request.urlopen(req, timeout=30) as r:
            o = json.load(r)
    k = o["validation"]["profiles"][PROFILE]["k"]
    # El canal solo publica BTC y ETH (el resto, en la web): 23 pares serían un mensaje enorme.
    results = [oracle(b, o["refs"][b + "USDT"], k) for b in CHANNEL_PAIRS]
    n_locked = len(o["locked"])
    now = datetime.now()
    today = now.date().isoformat()
    silent = now.hour < DAILY_HOUR or now.hour >= 23

    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    st.setdefault("dir", {})
    st.setdefault("pending", {})
    out: list[tuple[str, bool]] = []

    if a.init:
        # Instalación nueva: se da por enviado el resumen de hoy (si ya es la hora) y se
        # toma la dirección actual como referencia para los avisos. No se envía nada.
        st["dir"] = {r["base"]: r["dir"] for r in results}
        st["pending"] = {}
        if now.hour >= DAILY_HOUR:
            st["daily"] = today
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print("Listo. Estado guardado en", STATE, "·", st["dir"])
        return
    if a.daily or (st.get("daily") != today and now.hour >= DAILY_HOUR):
        out.append((daily_msg(results, n_locked, now), False))
        st["daily"] = today
        st["dir"] = {r["base"]: r["dir"] for r in results}
        st["pending"] = {}
    else:
        for r in results:
            b, old = r["base"], st["dir"].get(r["base"])
            if old is None:
                st["dir"][b] = r["dir"]
            elif r["dir"] == old:
                st["pending"].pop(b, None)
            elif st["pending"].get(b) == r["dir"]:  # segunda comprobación seguida: se confirma
                out.append((change_msg(r, old, n_locked), silent))
                st["dir"][b] = r["dir"]
                st["pending"].pop(b, None)
            else:
                st["pending"][b] = r["dir"]

    for text, quiet in out:
        print(text, end="\n\n")
        if not a.dry:
            send(text, quiet)
    if not out:
        print("Nada que publicar:", {r["base"]: r["dir"] for r in results}, "pendientes:", st["pending"])
    if not a.dry:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
