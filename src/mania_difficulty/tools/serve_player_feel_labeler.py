from __future__ import annotations

import argparse
import json
import numbers
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from mania_difficulty.player_feel_annotations import ANNOTATION_COLUMNS, ensure_annotation_columns


CHOICE_CANONICAL = {
    "a": "A",
    "left": "A",
    "1": "A",
    "b": "B",
    "right": "B",
    "2": "B",
    "tie": "tie",
    "same": "tie",
    "equal": "tie",
    "uncertain": "uncertain",
    "unknown": "uncertain",
    "out_of_range": "out_of_range",
    "range": "out_of_range",
    "skip": "skip",
    "": "",
}

SOFT_UNJUDGED = {"tie", "uncertain", "out_of_range", "skip"}


def clean_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_choice(value: object) -> str:
    if pd.isna(value):
        return ""
    return CHOICE_CANONICAL.get(str(value).strip().lower(), str(value).strip())


def label_status(row: dict[str, object] | pd.Series) -> str:
    choice = normalize_choice(row.get("harder_choice", ""))
    if choice in {"A", "B"}:
        return "judged"
    if choice in SOFT_UNJUDGED:
        return choice
    return "open"


def row_to_json(row: pd.Series) -> dict[str, object]:
    return {column: clean_value(row.get(column, "")) for column in ANNOTATION_COLUMNS}


def strain_values(frame: pd.DataFrame) -> list[float]:
    values = pd.concat(
        [
            pd.to_numeric(frame["a_peak_strain"], errors="coerce"),
            pd.to_numeric(frame["b_peak_strain"], errors="coerce"),
        ]
    ).dropna()
    return sorted(float(value) for value in values)


def percentile_rank(value: object, values: list[float]) -> float | None:
    if not values:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    below_or_equal = sum(1 for candidate in values if candidate <= numeric_value)
    return round((below_or_equal / len(values)) * 100.0, 1)


def add_strain_context(pair: dict[str, object] | None, values: list[float]) -> dict[str, object] | None:
    if pair is None:
        return None
    output = dict(pair)
    for prefix in ("a", "b"):
        output[f"{prefix}_strain_percentile"] = percentile_rank(
            output.get(f"{prefix}_peak_strain", ""), values
        )
    return output


def add_pair_peak(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    a_peak = pd.to_numeric(output["a_peak_strain"], errors="coerce").fillna(0.0)
    b_peak = pd.to_numeric(output["b_peak_strain"], errors="coerce").fillna(0.0)
    output["_pair_peak"] = pd.concat([a_peak, b_peak], axis=1).max(axis=1)
    return output


class LabelerStore:
    def __init__(self, pairs_csv: Path):
        self.pairs_csv = Path(pairs_csv)

    def read(self) -> pd.DataFrame:
        if not self.pairs_csv.exists():
            raise FileNotFoundError(f"No pairs CSV found: {self.pairs_csv}")
        return ensure_annotation_columns(pd.read_csv(self.pairs_csv).fillna(""))

    def write(self, frame: pd.DataFrame) -> None:
        self.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        output = ensure_annotation_columns(frame.fillna(""))
        temp_path = self.pairs_csv.with_suffix(self.pairs_csv.suffix + ".tmp")
        output.to_csv(temp_path, index=False, encoding="utf-8")
        temp_path.replace(self.pairs_csv)

    def state(
        self,
        *,
        stage: str = "",
        scope: str = "",
        status: str = "",
        index: int = 0,
    ) -> dict[str, object]:
        frame = self.read()
        frame["_status"] = [label_status(row) for _, row in frame.iterrows()]
        filtered = frame
        if stage:
            filtered = filtered[filtered["player_stage"].astype(str) == stage]
        if scope:
            filtered = filtered[filtered["scope"].astype(str) == scope]
        if status:
            filtered = filtered[filtered["_status"].astype(str) == status]
        filtered = add_pair_peak(filtered).sort_values("_pair_peak", ascending=False).reset_index(drop=True)
        safe_index = max(0, min(int(index), max(len(filtered) - 1, 0)))
        values = strain_values(frame)
        current = row_to_json(filtered.iloc[safe_index]) if len(filtered) else None
        current = add_strain_context(current, values)
        return {
            "pairs_csv": str(self.pairs_csv),
            "total_count": int(len(frame)),
            "filtered_count": int(len(filtered)),
            "index": safe_index,
            "current": current,
            "judged_count": int((frame["_status"] == "judged").sum()),
            "open_count": int((frame["_status"] == "open").sum()),
            "uncertain_count": int((frame["_status"] == "uncertain").sum()),
            "out_of_range_count": int((frame["_status"] == "out_of_range").sum()),
            "tie_count": int((frame["_status"] == "tie").sum()),
            "skip_count": int((frame["_status"] == "skip").sum()),
            "stages": sorted(str(value) for value in frame["player_stage"].dropna().unique()),
            "scopes": sorted(str(value) for value in frame["scope"].dropna().unique()),
            "strain_reference": {
                "min": round(values[0], 3) if values else None,
                "max": round(values[-1], 3) if values else None,
            },
        }

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        pair_id = str(payload.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("pair_id is required.")
        frame = self.read()
        matches = frame.index[frame["pair_id"].astype(str) == pair_id].tolist()
        if not matches:
            raise KeyError(f"Unknown pair_id: {pair_id}")
        row_index = matches[0]
        frame.loc[row_index, "harder_choice"] = normalize_choice(payload.get("harder_choice", ""))
        for column in ("confidence", "reason_tags", "notes"):
            if column in payload:
                frame.loc[row_index, column] = "" if payload[column] is None else str(payload[column])
        self.write(frame)
        saved_row = row_to_json(frame.loc[row_index])
        return {"status": "ok", "pair": saved_row, "row_status": label_status(saved_row)}

    def save_json(self, data: bytes) -> dict[str, object]:
        return self.save(json.loads(data.decode("utf-8")))


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>4K 體感標註</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #181c22;
      --panel-2: #20262e;
      --line: #303947;
      --text: #eef2f7;
      --muted: #9aa7b8;
      --accent: #78d0ff;
      --good: #7ddc9a;
      --warn: #f2c36b;
      --bad: #ff8b8b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, system-ui, sans-serif;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #11151bcc;
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .stat { color: var(--muted); font-size: 13px; }
    main {
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .toolbar, .judgebar {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    label { color: var(--muted); font-size: 12px; display: grid; gap: 4px; }
    select, input, textarea, button {
      background: var(--panel-2);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      min-height: 38px;
      font-weight: 650;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: #11445d; border-color: #1f83ad; }
    button.good { background: #143d27; border-color: #2f8c55; }
    button.warn { background: #443415; border-color: #9d7424; }
    button.bad { background: #4a2020; border-color: #9d4141; }
    .comparison {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
    }
    .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 300px;
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .side h2 {
      margin: 0;
      font-size: 26px;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      font-size: 12px;
      gap: 6px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      background: #12161c;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
    }
    .metric span { display: block; color: var(--muted); font-size: 11px; }
    .metric strong { display: block; margin-top: 3px; font-size: 18px; overflow-wrap: anywhere; }
    .metric small { display: block; margin-top: 4px; color: var(--muted); font-size: 11px; }
    .strain-bar {
      margin-top: 8px;
      height: 6px;
      width: 100%;
      overflow: hidden;
      border-radius: 999px;
      background: #2d3541;
    }
    .strain-bar i {
      display: block;
      height: 100%;
      width: var(--fill);
      background: linear-gradient(90deg, var(--good), var(--warn), var(--bad));
    }
    .judgebar { align-items: end; }
    .choice-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      flex: 1 1 520px;
    }
    .notes {
      display: grid;
      grid-template-columns: minmax(120px, 180px) minmax(180px, 280px) minmax(240px, 1fr);
      gap: 10px;
      width: 100%;
    }
    textarea { min-height: 76px; resize: vertical; }
    .footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .error { color: var(--bad); }
    @media (max-width: 850px) {
      .comparison { grid-template-columns: 1fr; }
      .notes { grid-template-columns: 1fr; }
      header { position: static; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>4K 體感標註</h1>
    <div id="stats" class="stat"></div>
  </header>
  <main>
    <section class="toolbar">
      <label>階段<select id="stage"></select></label>
      <label>範圍<select id="scope"></select></label>
      <label>狀態<select id="status">
        <option value="open">open</option>
        <option value="">all</option>
        <option value="judged">judged</option>
        <option value="uncertain">uncertain</option>
        <option value="out_of_range">out_of_range</option>
        <option value="tie">tie</option>
        <option value="skip">skip</option>
      </select></label>
      <button id="prev">上一題</button>
      <button id="next">下一題</button>
      <button id="reload">重整</button>
    </section>
    <section id="comparison" class="comparison"></section>
    <section class="judgebar">
      <div class="choice-row">
        <button class="good" data-choice="A">A 較難</button>
        <button class="good" data-choice="B">B 較難</button>
        <button class="warn" data-choice="tie">差不多</button>
        <button class="warn" data-choice="uncertain">不確定</button>
        <button class="bad" data-choice="out_of_range">超出體感</button>
        <button data-choice="skip">跳過</button>
      </div>
      <div class="notes">
        <label>信心<input id="confidence" type="number" min="0" max="5" step="0.5" value="1"></label>
        <label>原因標籤<input id="reason_tags" placeholder="jack,reading,ln"></label>
        <label>備註<textarea id="notes"></textarea></label>
      </div>
    </section>
    <section class="footer">
      <span id="pairMeta"></span>
      <span id="message"></span>
    </section>
  </main>
  <script>
    const state = { stage: "dan_ready", scope: "", status: "open", index: 0, current: null };
    const $ = (id) => document.getElementById(id);
    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function fmt(value) {
      const n = Number(value);
      if (Number.isFinite(n)) return n.toFixed(2);
      return value || "";
    }
    function pressureBand(percentile) {
      const n = Number(percentile);
      if (!Number.isFinite(n)) return "未知";
      if (n < 20) return "低";
      if (n < 40) return "偏低";
      if (n < 60) return "中";
      if (n < 80) return "偏高";
      return "高";
    }
    function pressureMetric(pair, prefix) {
      const raw = pair[`${prefix}_peak_strain`];
      const percentile = Number(pair[`${prefix}_strain_percentile`]);
      const safePercentile = Number.isFinite(percentile) ? Math.max(0, Math.min(100, percentile)) : 0;
      return `<div class="metric">
        <span>相對壓力</span>
        <strong>${esc(pressureBand(percentile))}</strong>
        <small>PR ${Number.isFinite(percentile) ? percentile.toFixed(1) : "?"} · 原始 ${esc(fmt(raw))}</small>
        <div class="strain-bar"><i style="--fill: ${safePercentile}%"></i></div>
      </div>`;
    }
    function side(pair, prefix, label) {
      const title = pair[`${prefix}_title`] || `Beatmap ${pair[`${prefix}_beatmap_id`]}`;
      const segment = pair[`${prefix}_start_sec`] !== "" ? `${fmt(pair[`${prefix}_start_sec`])}s - ${fmt(pair[`${prefix}_end_sec`])}s` : "整張";
      return `<article class="side">
        <span class="badge">${label} · ${esc(pair[`${prefix}_dominant_skill`] || "unknown")} · ${esc(segment)}</span>
        <h2>${esc(title)}</h2>
        <div class="stat">${esc(pair[`${prefix}_artist`] || "")} ${esc(pair[`${prefix}_version`] || "")}</div>
        <div class="metric-grid">
          <div class="metric"><span>Beatmap</span><strong>${esc(pair[`${prefix}_beatmap_id`])}</strong></div>
          ${pressureMetric(pair, prefix)}
          <div class="metric"><span>主技能</span><strong>${esc(pair[`${prefix}_dominant_skill`] || "")}</strong></div>
        </div>
      </article>`;
    }
    function fillSelect(select, values, current, allLabel) {
      const existing = select.value;
      select.innerHTML = `<option value="">${allLabel}</option>` + values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
      select.value = current ?? existing ?? "";
    }
    async function load() {
      const params = new URLSearchParams({ stage: state.stage, scope: state.scope, status: state.status, index: state.index });
      const res = await fetch(`/api/state?${params}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "load failed");
      fillSelect($("stage"), data.stages, state.stage, "all");
      fillSelect($("scope"), data.scopes, state.scope, "all");
      $("status").value = state.status;
      state.index = data.index;
      state.current = data.current;
      $("stats").textContent = `${data.judged_count}/${data.total_count} 已判斷 · ${data.open_count} 未標 · ${data.out_of_range_count} 超出體感`;
      if (!data.current) {
        $("comparison").innerHTML = `<article class="side"><h2>沒有題目</h2><div class="stat">換個篩選或重整看看。</div></article>`;
        $("pairMeta").textContent = "";
        return;
      }
      const pair = data.current;
      $("comparison").innerHTML = side(pair, "a", "A") + side(pair, "b", "B");
      $("confidence").value = pair.confidence || 1;
      $("reason_tags").value = pair.reason_tags || "";
      $("notes").value = pair.notes || "";
      $("pairMeta").textContent = `${pair.pair_id} · ${pair.scope} · ${pair.player_stage} · ${data.index + 1}/${data.filtered_count}`;
    }
    async function save(choice) {
      if (!state.current) return;
      $("message").textContent = "儲存中...";
      const payload = {
        pair_id: state.current.pair_id,
        harder_choice: choice,
        confidence: $("confidence").value,
        reason_tags: $("reason_tags").value,
        notes: $("notes").value
      };
      const res = await fetch("/api/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) {
        $("message").innerHTML = `<span class="error">${esc(data.error || "save failed")}</span>`;
        return;
      }
      $("message").textContent = `已儲存 ${choice}`;
      await load();
    }
    $("stage").addEventListener("change", e => { state.stage = e.target.value; state.index = 0; load(); });
    $("scope").addEventListener("change", e => { state.scope = e.target.value; state.index = 0; load(); });
    $("status").addEventListener("change", e => { state.status = e.target.value; state.index = 0; load(); });
    $("prev").addEventListener("click", () => { state.index = Math.max(0, state.index - 1); load(); });
    $("next").addEventListener("click", () => { state.index += 1; load(); });
    $("reload").addEventListener("click", () => load());
    document.querySelectorAll("[data-choice]").forEach(btn => btn.addEventListener("click", () => save(btn.dataset.choice)));
    document.addEventListener("keydown", (event) => {
      if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
      const key = event.key.toLowerCase();
      if (key === "a") save("A");
      if (key === "b") save("B");
      if (key === "u") save("uncertain");
      if (key === "o") save("out_of_range");
      if (key === "s") save("skip");
      if (key === "arrowright") { state.index += 1; load(); }
      if (key === "arrowleft") { state.index = Math.max(0, state.index - 1); load(); }
    });
    load().catch(err => $("message").innerHTML = `<span class="error">${esc(err.message)}</span>`);
  </script>
</body>
</html>
"""


class LabelerHandler(BaseHTTPRequestHandler):
    store: LabelerStore

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html()
                return
            if parsed.path == "/health":
                self._send_json({"status": "ok", "pairs_csv": str(self.store.pairs_csv)})
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                state = self.store.state(
                    stage=query.get("stage", [""])[0],
                    scope=query.get("scope", [""])[0],
                    status=query.get("status", [""])[0],
                    index=int(query.get("index", ["0"])[0] or 0),
                )
                self._send_json(state)
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            result = self.store.save_json(self.rfile.read(length))
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve_labeler(pairs_csv: Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(LabelerHandler):
        store = LabelerStore(pairs_csv)

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local 4K player-feel annotation UI.")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("outputs/player_feel_v1_pilot_real/player_feel_pairs_to_label.csv"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = serve_labeler(args.pairs, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving player-feel labeler at {url}")
    print(f"Pairs CSV: {args.pairs}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
