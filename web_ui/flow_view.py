"""Clickable execution-flow diagram for one answer (self-contained SVG + JS).

The page receives the turn's workflow_trace (see agents/planned_agent/trace.py)
and renders a fixed pipeline. Clicking a node shows every attempt of that step
with its real input/output; all interaction runs in the browser, so Streamlit
does not rerun.
"""

import json

PIPELINES = {
    "qwen27b": {
        "nodes": [
            ("agent_sql", "寫 SQL"),
            ("step_run_sql", "唯讀執行 SQL"),
            ("agent_review", "語意審查"),
            ("agent_report", "寫解釋與圖表規格"),
            ("step_publish", "發布檢查"),
            ("step_draw_chart", "Plotly 畫圖"),
        ],
        "loops": [
            {"from": "agent_review", "to": "agent_sql", "label": "SQL 錯誤或審查拒絕 → 重寫（最多 3 次）"},
            {"from": "step_publish", "to": "agent_report", "label": "規格不合法 → 重寫（最多 2 次）"},
        ],
    },
}

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root{--bg:#ffffff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--panel:#f6f8fa;
--agent:#ede7fb;--agent-b:#7c5cd6;--step:#e8f0fe;--step-b:#3b6fd9;--tool:#fff4e0;--tool-b:#c27c0e;
--ok:#1a7f37;--bad:#cf222e;--skip:#8c959f;--sel:#0969da}
@media (prefers-color-scheme: dark){:root{--bg:#0e1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--panel:#161b22;
--agent:#2a2143;--agent-b:#a78bfa;--step:#16253f;--step-b:#79a6f2;--tool:#33270f;--tool-b:#e3a43a;
--ok:#3fb950;--bad:#f85149;--skip:#6e7681;--sel:#58a6ff}}
*{box-sizing:border-box}
body{margin:0;font-family:"Source Sans Pro",system-ui,-apple-system,"Noto Sans TC",sans-serif;color:var(--fg);background:var(--bg);font-size:14px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:2px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:18px;height:12px;border:1.5px solid;display:inline-block}
.scroller{overflow-x:auto}
svg{width:100%;min-width:720px;height:auto;display:block}
.node{cursor:pointer}
.node:focus{outline:none}
.node:focus .box,.node.sel .box{stroke:var(--sel);stroke-width:3}
.name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:600;fill:var(--fg)}
.desc{font-size:12px;fill:var(--muted)}
.loop-label{font-size:11px;fill:var(--muted)}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 8px}
.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;cursor:pointer;background:var(--panel);color:var(--fg);font-family:ui-monospace,Menlo,monospace}
.chip.sel{border-color:var(--sel);box-shadow:0 0 0 1px var(--sel)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
.panel{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:10px 12px}
.panel h4{margin:0 0 6px;font-size:14px}
.meta{color:var(--muted);font-size:12px;margin-bottom:8px}
.io{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:640px){.io{grid-template-columns:1fr}}
.io h5{margin:0 0 4px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:260px;overflow:auto;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px}
table{border-collapse:collapse;font-size:12px;margin:4px 0;background:var(--bg)}
td,th{border:1px solid var(--line);padding:2px 6px;text-align:left}
.tablewrap{max-height:200px;overflow:auto}
.empty{color:var(--muted);padding:8px 0}
.key{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);margin:6px 0 2px}
</style></head><body>
<div class="legend">
  <span><i class="sw" style="background:var(--agent);border-color:var(--agent-b);border-radius:8px"></i>agent_：LLM 判斷</span>
  <span><i class="sw" style="background:var(--step);border-color:var(--step-b)"></i>step_：程式，一定執行</span>
  <span><i class="sw" style="background:var(--tool);border-color:var(--tool-b);border-style:dashed"></i>tool_：模型自己呼叫</span>
  <span><i class="dot" style="background:var(--ok)"></i>通過</span>
  <span><i class="dot" style="background:var(--bad)"></i>錯誤／拒絕</span>
  <span><i class="dot" style="background:var(--skip)"></i>未執行／略過</span>
</div>
<div class="scroller"><svg id="flow" role="img" aria-label="執行流程圖"></svg></div>
<div class="chips" id="chips"></div>
<div class="panel" id="panel"><div class="empty">點流程圖中的任一步驟，查看它的輸入與輸出。</div></div>
<script>
const TRACE = __TRACE__;
const PIPE = __PIPE__;
const NS = "http://www.w3.org/2000/svg";
const KIND = {agent:"LLM agent", step:"程式步驟", tool:"Tool（模型呼叫）"};
const STATUS = {ok:"通過", rejected:"審查拒絕", error:"錯誤", skipped:"略過", running:"執行中"};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const kindOf = n => n.split("_")[0];
const W = 132, H = 58, GAP = 40, X0 = 20, Y0 = 70;
const svg = document.getElementById("flow");
const total = X0*2 + PIPE.nodes.length*W + (PIPE.nodes.length-1)*GAP;
svg.setAttribute("viewBox", `0 0 ${total} 156`);
svg.style.maxWidth = `${total}px`;
const pos = {};
PIPE.nodes.forEach(([n], i) => pos[n] = {x: X0 + i*(W+GAP), y: Y0});

function el(tag, attrs, parent){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k, attrs[k]); (parent||svg).appendChild(e); return e; }
function text(str, attrs, parent){ const t=el("text", attrs, parent); t.textContent=str; return t; }

const defs = el("defs", {});
const mk = el("marker", {id:"arr", viewBox:"0 0 10 10", refX:"9", refY:"5", markerWidth:"7", markerHeight:"7", orient:"auto-start-reverse"}, defs);
el("path", {d:"M0,0 L10,5 L0,10 z", fill:css("--muted")}, mk);

for(let i=0;i<PIPE.nodes.length-1;i++){
  const a = pos[PIPE.nodes[i][0]], b = pos[PIPE.nodes[i+1][0]];
  el("line", {x1:a.x+W, y1:a.y+H/2, x2:b.x-2, y2:b.y+H/2, stroke:css("--muted"), "stroke-width":1.5, "marker-end":"url(#arr)"});
}
PIPE.loops.forEach((lp, i) => {
  const a = pos[lp.from], b = pos[lp.to];
  const x1 = a.x + W/2, x2 = b.x + W/2, y = Y0 - 6, top = 22 + i*0;
  el("path", {d:`M${x1},${y} C${x1},${top} ${x2},${top} ${x2},${y-2}`, fill:"none", stroke:css("--muted"), "stroke-width":1.2, "stroke-dasharray":"4 3", "marker-end":"url(#arr)"});
  text(lp.label, {x:(x1+x2)/2, y:top+6, "text-anchor":"middle", class:"loop-label"});
});

const runs = {};
TRACE.forEach((e, i) => { (runs[e.name] = runs[e.name] || []).push(i); });
function nodeStatus(name){
  const r = runs[name]; if(!r) return {color:css("--skip"), label:"未執行", count:0, hadFail:false};
  const last = TRACE[r[r.length-1]];
  const hadFail = r.some(i => ["error","rejected"].includes(TRACE[i].status));
  const c = last.status==="ok" ? css("--ok") : (last.status==="skipped" ? css("--skip") : css("--bad"));
  return {color:c, label:STATUS[last.status]||last.status, count:r.length, hadFail};
}

let selected = null, selectedEntry = null;
const nodeEls = {};
PIPE.nodes.forEach(([name, desc]) => {
  const p = pos[name], k = kindOf(name), st = nodeStatus(name);
  const g = el("g", {class:"node", tabindex:"0", role:"button", "aria-label":`${name} ${desc} ${st.label}`});
  const fill = css(`--${k}`), stroke = css(`--${k}-b`);
  const box = el("rect", {class:"box", x:p.x, y:p.y, width:W, height:H, rx: k==="agent"?16:3, fill, stroke, "stroke-width":1.6,
    "stroke-dasharray": k==="tool"?"5 3":"none", opacity: st.count? 1 : 0.55}, g);
  text(name, {x:p.x+W/2, y:p.y+24, "text-anchor":"middle", class:"name"}, g);
  text(desc, {x:p.x+W/2, y:p.y+43, "text-anchor":"middle", class:"desc"}, g);
  el("circle", {cx:p.x+W-10, cy:p.y+10, r:5, fill:st.color}, g);
  if(st.count>1) text(`×${st.count}`, {x:p.x+W-4, y:p.y+H+16, "text-anchor":"end", class:"desc", style:`fill:${st.hadFail?css("--bad"):css("--muted")}`}, g);
  const ms = (runs[name]||[]).reduce((s,i)=>s+(TRACE[i].ms||0),0);
  if(ms) text(`${(ms/1000).toFixed(1)} 秒`, {x:p.x+4, y:p.y+H+16, class:"desc"}, g);
  g.addEventListener("click", () => select(name));
  g.addEventListener("keydown", ev => { if(ev.key==="Enter"||ev.key===" "){ ev.preventDefault(); select(name); }});
  nodeEls[name] = g;
});

function renderValue(v){
  const wrap = document.createElement("div");
  if(v && typeof v==="object" && Array.isArray(v.columns) && Array.isArray(v.rows)){
    const rest = Object.assign({}, v); delete rest.columns; delete rest.rows;
    const meta = document.createElement("div"); meta.className="meta";
    meta.textContent = `共 ${v.row_count ?? v.rows.length} 列` + (v.rows_shown!=null && v.rows_shown < v.row_count ? `，顯示前 ${v.rows_shown} 列` : "");
    wrap.appendChild(meta);
    const tw = document.createElement("div"); tw.className="tablewrap";
    const t = document.createElement("table"); const hr = t.insertRow();
    v.columns.forEach(c => { const th=document.createElement("th"); th.textContent=c; hr.appendChild(th); });
    v.rows.forEach(r => { const tr=t.insertRow(); r.forEach(c => { tr.insertCell().textContent = c===null?"—":String(c); }); });
    tw.appendChild(t); wrap.appendChild(tw);
    delete rest.row_count; delete rest.rows_shown;
    if(Object.keys(rest).length){ wrap.appendChild(renderValue(rest)); }
    return wrap;
  }
  if(v && typeof v==="object" && !Array.isArray(v)){
    Object.entries(v).forEach(([k, item]) => {
      const lab = document.createElement("div"); lab.className="key"; lab.textContent = k; wrap.appendChild(lab);
      if(item && typeof item==="object") wrap.appendChild(renderValue(item));
      else { const pre=document.createElement("pre"); pre.textContent = (item===""||item==null) ? "（空）" : String(item); wrap.appendChild(pre); }
    });
    return wrap;
  }
  const pre = document.createElement("pre");
  pre.textContent = v==null ? "（無）" : JSON.stringify(v, null, 2);
  wrap.appendChild(pre); return wrap;
}

function showEntry(i){
  selectedEntry = i;
  const e = TRACE[i], panel = document.getElementById("panel");
  panel.replaceChildren();
  const h = document.createElement("h4"); h.textContent = `${e.name}（第 ${e.attempt} 次）`; panel.appendChild(h);
  const m = document.createElement("div"); m.className="meta";
  m.textContent = `${KIND[e.kind]||e.kind}　·　${STATUS[e.status]||e.status}` + (e.ms!=null ? `　·　${(e.ms/1000).toFixed(2)} 秒` : "　·　在上一步內執行");
  panel.appendChild(m);
  const io = document.createElement("div"); io.className="io";
  [["輸入", e.input], ["輸出", e.output]].forEach(([label, v]) => {
    const col = document.createElement("div"); const h5=document.createElement("h5"); h5.textContent=label;
    col.appendChild(h5); col.appendChild(renderValue(v)); io.appendChild(col);
  });
  panel.appendChild(io);
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("sel", Number(c.dataset.i)===i));
}

function select(name){
  selected = name;
  Object.entries(nodeEls).forEach(([n,g]) => g.classList.toggle("sel", n===name));
  const chips = document.getElementById("chips"); chips.replaceChildren();
  const r = runs[name] || [];
  if(!r.length){
    const panel = document.getElementById("panel"); panel.replaceChildren();
    const d=document.createElement("div"); d.className="empty"; d.textContent=`${name} 這次沒有執行（前面的步驟沒有通過，或流程提早結束）。`; panel.appendChild(d); return;
  }
  r.forEach(i => {
    const e = TRACE[i], b = document.createElement("button"); b.className="chip"; b.dataset.i=i;
    const dot=document.createElement("i"); dot.className="dot";
    dot.style.background = e.status==="ok"?css("--ok"):(e.status==="skipped"?css("--skip"):css("--bad"));
    b.appendChild(dot); b.appendChild(document.createTextNode(`第 ${e.attempt} 次 · ${STATUS[e.status]||e.status}`));
    b.addEventListener("click", () => showEntry(i)); chips.appendChild(b);
  });
  showEntry(r[r.length-1]);
}

if(TRACE.length){ select(TRACE[TRACE.length-1].name); }
else { document.getElementById("panel").firstElementChild.textContent = "這一則沒有執行紀錄（例如問候語，不進入查詢流程）。"; }
</script></body></html>"""


def flow_html(trace: list[dict], backend: str) -> str:
    pipe = PIPELINES.get(backend) or PIPELINES["qwen27b"]
    safe = lambda value: json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__TRACE__", safe(trace or [])).replace("__PIPE__", safe(pipe))


def render_flow(trace: list[dict], backend: str, height: int = 640) -> None:
    import streamlit.components.v1 as components

    components.html(flow_html(trace, backend), height=height, scrolling=True)
