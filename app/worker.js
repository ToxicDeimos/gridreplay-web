// Motor de GridReplay en el navegador: ejecuta gridreplay/web.py con Pyodide.
// Sirve al analizador (el export de Pionex llega por postMessage y no sale de aquí) y al simulador
// de robustez. Solo se descargan precios públicos de Binance.
// Mensajes al hilo principal: {type: "stage" | "download" | "progress" | "done" | "error", ...}.

// Servido desde la propia web (build.py): los CDN públicos caen con los bloqueos de LaLiga.
const PYODIDE = "/pyodide/";
const MODULES = ["__init__", "data", "fees", "pionex", "robust", "sim", "validate", "web"];
const VERSION = "d9a0da9ef7";
const MAX_PARALLEL = 6;

importScripts(PYODIDE + "pyodide.js");

const post = (msg) => self.postMessage(msg);

// fetch con límite de concurrencia y reintentos; devuelve [estado, bytes].
let active = 0;
const queue = [];
let files = 0;
let bytes = 0;
async function slot() {
  if (active < MAX_PARALLEL) return void active++;
  await new Promise((resolve) => queue.push(resolve));
}
function release() {
  const next = queue.shift();
  next ? next() : active--;
}
async function get(url) {
  await slot();
  try {
    for (let attempt = 0; ; attempt++) {
      try {
        const resp = await fetch(url);
        if ((resp.status === 429 || resp.status >= 500) && attempt < 3) throw new Error(`HTTP ${resp.status}`);
        const body = new Uint8Array(await resp.arrayBuffer());
        files++;
        bytes += body.length;
        post({ type: "download", files, mb: bytes / 1e6 });
        return [resp.status, body];
      } catch (err) {
        if (attempt >= 3) throw err;
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  } finally {
    release();
  }
}

let ready = null;
async function boot() {
  post({ type: "stage", stage: "engine" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "pandas"]);
  py.FS.mkdirTree("/app/gridreplay");
  await Promise.all(
    MODULES.map(async (m) => {
      const resp = await fetch(`/app/py/gridreplay/${m}.py?v=${VERSION}`);
      if (!resp.ok) throw new Error(`No se pudo cargar ${m}.py`);
      py.FS.writeFile(`/app/gridreplay/${m}.py`, await resp.text());
    }),
  );
  py.globals.set("js_get", get);
  py.globals.set("js_progress", (s) => post({ type: "progress", ...JSON.parse(s) }));
  // data.py importa requests para el modo de escritorio; aquí no se usa.
  await py.runPythonAsync(`
import sys, types, json
sys.path.insert(0, "/app")
sys.modules.setdefault("requests", types.ModuleType("requests"))
from gridreplay.web import analyze, robustness, shareable

async def get(url):
    try:
        status, body = await js_get(url)
    except Exception as exc:
        raise OSError(f"Descarga fallida: {url}") from exc
    return int(status), body.to_bytes()

async def run(export):
    import zipfile
    try:
        rows = await analyze(export.to_bytes(), get, lambda d: js_progress(json.dumps(d)))
    except (KeyError, zipfile.BadZipFile):
        return json.dumps({"error": "not_export"})
    return json.dumps({"rows": rows, "share": shareable(rows)})

async def robust(params):
    out = await robustness(json.loads(params), get, lambda d: js_progress(json.dumps(d)))
    return json.dumps(out)
`);
  return py;
}

const fail = (err) => post({ type: "error", message: String(err?.message ?? err).slice(-600) });

self.onmessage = async ({ data }) => {
  let py;
  try {
    ready ??= boot();
    py = await ready;
  } catch (err) {
    ready = null; // se puede reintentar (p. ej. tras perder la conexión)
    return fail(err);
  }
  if (data.type === "warmup") return post({ type: "stage", stage: "ready" });
  try {
    files = 0;
    bytes = 0;
    let out;
    if (data.type === "robust") {
      post({ type: "stage", stage: "prices" });
      out = JSON.parse(await py.globals.get("robust")(JSON.stringify(data.params)));
    } else {
      post({ type: "stage", stage: "reading" });
      out = JSON.parse(await py.globals.get("run")(new Uint8Array(data.file)));
    }
    post(out.error ? { type: "error", code: out.error, symbol: out.symbol } : { type: "done", ...out });
  } catch (err) {
    fail(err);
  }
};
