"""
Live Training Dashboard
========================
Serves a real-time web dashboard that monitors YOLO training progress
by reading results.csv every 5 seconds.

Usage:
    .venv\\Scripts\\python.exe training/dashboard.py
    Then open: http://localhost:8080
"""

import os
import csv
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_CSV = BASE_DIR / "results" / "enhanced-uwear-dev" / "results.csv"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Enhanced UWear — Live Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 20px;
  }
  h1 {
    text-align: center;
    font-size: 1.6rem;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }
  .status-bar {
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 20px;
  }
  .status-bar .live { color: #22c55e; font-weight: bold; }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }
  .card .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
  .card .sub { font-size: 0.7rem; color: #666; margin-top: 2px; }
  .card.highlight { border-color: #6366f1; }
  .card .value.good { color: #22c55e; }
  .card .value.warn { color: #f59e0b; }
  .card .value.bad { color: #ef4444; }
  .charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .chart-box {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px;
  }
  .chart-box h3 { font-size: 0.9rem; color: #aaa; margin-bottom: 8px; }
  canvas { width: 100% !important; }
  .baseline { color: #f59e0b; font-size: 0.75rem; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>Enhanced UWear — Training Dashboard</h1>
<div class="status-bar">
  <span class="live">● LIVE</span> &nbsp; Auto-refreshing every 5s &nbsp;|&nbsp;
  Epoch <span id="epoch">-</span>/100 &nbsp;|&nbsp;
  Last update: <span id="lastUpdate">-</span>
</div>

<div class="cards">
  <div class="card highlight">
    <div class="label">mAP@0.5</div>
    <div class="value" id="map50">-</div>
    <div class="sub baseline">UWear: 0.7883</div>
  </div>
  <div class="card">
    <div class="label">Precision</div>
    <div class="value" id="precision">-</div>
    <div class="sub baseline">UWear: 0.7336</div>
  </div>
  <div class="card">
    <div class="label">Recall</div>
    <div class="value" id="recall">-</div>
    <div class="sub baseline">UWear: 0.7881</div>
  </div>
  <div class="card">
    <div class="label">mAP@50-95</div>
    <div class="value" id="map5095">-</div>
    <div class="sub">Higher = better</div>
  </div>
  <div class="card">
    <div class="label">Box Loss</div>
    <div class="value" id="boxloss">-</div>
    <div class="sub">Lower = better</div>
  </div>
  <div class="card">
    <div class="label">Cls Loss</div>
    <div class="value" id="clsloss">-</div>
    <div class="sub">Lower = better</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>mAP@0.5 vs UWear Baseline</h3>
    <canvas id="mapChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Training Losses</h3>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Precision & Recall</h3>
    <canvas id="prChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Validation Losses</h3>
    <canvas id="valLossChart"></canvas>
  </div>
</div>

<script>
const UWEAR_MAP = 0.7883;
const UWEAR_P = 0.7336;
const UWEAR_R = 0.7881;

const chartOpts = (yLabel) => ({
  responsive: true,
  animation: { duration: 300 },
  scales: {
    x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#222' } },
    y: { title: { display: true, text: yLabel, color: '#888' }, ticks: { color: '#666' }, grid: { color: '#222' } }
  },
  plugins: { legend: { labels: { color: '#ccc', font: { size: 11 } } } }
});

const mapChart = new Chart(document.getElementById('mapChart'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'mAP@0.5', data: [], borderColor: '#6366f1', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
    { label: 'UWear Baseline', data: [], borderColor: '#f59e0b', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 }
  ]}, options: chartOpts('mAP@0.5')
});

const lossChart = new Chart(document.getElementById('lossChart'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Box Loss', data: [], borderColor: '#22c55e', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 },
    { label: 'Cls Loss', data: [], borderColor: '#ef4444', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 },
    { label: 'DFL Loss', data: [], borderColor: '#3b82f6', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 }
  ]}, options: chartOpts('Loss')
});

const prChart = new Chart(document.getElementById('prChart'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Precision', data: [], borderColor: '#a855f7', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
    { label: 'Recall', data: [], borderColor: '#06b6d4', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
    { label: 'UWear P', data: [], borderColor: '#a855f7', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 },
    { label: 'UWear R', data: [], borderColor: '#06b6d4', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 }
  ]}, options: chartOpts('Score')
});

const valLossChart = new Chart(document.getElementById('valLossChart'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Val Box', data: [], borderColor: '#22c55e', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 },
    { label: 'Val Cls', data: [], borderColor: '#ef4444', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 },
    { label: 'Val DFL', data: [], borderColor: '#3b82f6', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 2 }
  ]}, options: chartOpts('Validation Loss')
});

function colorize(el, val, baseline) {
  el.classList.remove('good','warn','bad');
  if (val >= baseline) el.classList.add('good');
  else if (val >= baseline * 0.9) el.classList.add('warn');
  else el.classList.add('bad');
}

async function refresh() {
  try {
    const res = await fetch('/api/results?' + Date.now());
    const rows = await res.json();
    if (!rows.length) return;

    const last = rows[rows.length - 1];
    const epochs = rows.map(r => r.epoch);

    document.getElementById('epoch').textContent = last.epoch;
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();

    const map50El = document.getElementById('map50');
    map50El.textContent = last.map50.toFixed(4);
    colorize(map50El, last.map50, UWEAR_MAP);

    const pEl = document.getElementById('precision');
    pEl.textContent = last.precision.toFixed(4);
    colorize(pEl, last.precision, UWEAR_P);

    const rEl = document.getElementById('recall');
    rEl.textContent = last.recall.toFixed(4);
    colorize(rEl, last.recall, UWEAR_R);

    document.getElementById('map5095').textContent = last.map5095.toFixed(4);
    document.getElementById('boxloss').textContent = last.box_loss.toFixed(4);
    document.getElementById('clsloss').textContent = last.cls_loss.toFixed(4);

    // Update charts
    mapChart.data.labels = epochs;
    mapChart.data.datasets[0].data = rows.map(r => r.map50);
    mapChart.data.datasets[1].data = epochs.map(() => UWEAR_MAP);
    mapChart.update();

    lossChart.data.labels = epochs;
    lossChart.data.datasets[0].data = rows.map(r => r.box_loss);
    lossChart.data.datasets[1].data = rows.map(r => r.cls_loss);
    lossChart.data.datasets[2].data = rows.map(r => r.dfl_loss);
    lossChart.update();

    prChart.data.labels = epochs;
    prChart.data.datasets[0].data = rows.map(r => r.precision);
    prChart.data.datasets[1].data = rows.map(r => r.recall);
    prChart.data.datasets[2].data = epochs.map(() => UWEAR_P);
    prChart.data.datasets[3].data = epochs.map(() => UWEAR_R);
    prChart.update();

    valLossChart.data.labels = epochs;
    valLossChart.data.datasets[0].data = rows.map(r => r.val_box);
    valLossChart.data.datasets[1].data = rows.map(r => r.val_cls);
    valLossChart.data.datasets[2].data = rows.map(r => r.val_dfl);
    valLossChart.update();
  } catch(e) { console.error(e); }
}

setInterval(refresh, 5000);
refresh();
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path.startswith('/api/results'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(read_results()).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress access logs


def read_results():
    if not RESULTS_CSV.exists():
        return []

    rows = []
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.reader(f)
        headers = None
        for row in reader:
            if headers is None:
                headers = [h.strip() for h in row]
                continue
            if len(row) < 12:
                continue
            try:
                vals = [v.strip() for v in row]
                rows.append({
                    'epoch': int(float(vals[0])),
                    'time': float(vals[1]),
                    'box_loss': float(vals[2]),
                    'cls_loss': float(vals[3]),
                    'dfl_loss': float(vals[4]),
                    'precision': float(vals[5]),
                    'recall': float(vals[6]),
                    'map50': float(vals[7]),
                    'map5095': float(vals[8]),
                    'val_box': float(vals[9]),
                    'val_cls': float(vals[10]),
                    'val_dfl': float(vals[11]),
                })
            except (ValueError, IndexError):
                continue
    return rows


if __name__ == '__main__':
    print("=" * 50)
    print("LIVE TRAINING DASHBOARD")
    print("=" * 50)
    print(f"Reading: {RESULTS_CSV}")
    print(f"Open:    http://localhost:8080")
    print("=" * 50)

    server = HTTPServer(('localhost', 8080), DashboardHandler)
    server.serve_forever()
