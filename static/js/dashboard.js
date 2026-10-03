(function () {
  "use strict";

  const PALETTE = ["#22d3ee", "#f43f5e", "#a78bfa", "#22c55e", "#f59e0b", "#38bdf8", "#fb7185", "#34d399"];
  const GRID = "rgba(148,163,184,0.14)";
  const TEXT = "#8fa3bf";

  Chart.defaults.color = TEXT;
  Chart.defaults.font.family = '"Segoe UI", Inter, system-ui, sans-serif';
  Chart.defaults.font.size = 12;

  function readPayload() {
    const node = document.getElementById("chart-data");
    if (!node) {
      return null;
    }
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      return null;
    }
  }

  function baseOptions(extra) {
    return Object.assign(
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: TEXT, boxWidth: 12 } },
          tooltip: { backgroundColor: "#0b1120", borderColor: "#22d3ee", borderWidth: 1 }
        },
        scales: {
          x: { ticks: { color: TEXT }, grid: { color: GRID } },
          y: { ticks: { color: TEXT }, grid: { color: GRID }, beginAtZero: true }
        }
      },
      extra || {}
    );
  }

  function emptyState(canvas, message) {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = TEXT;
    context.font = "13px Segoe UI";
    context.textAlign = "center";
    context.fillText(message, canvas.width / 2, canvas.height / 2);
  }

  function toEntries(mapping) {
    return Object.keys(mapping || {}).map((key) => ({ label: key, value: mapping[key] }));
  }

  function renderNormalMalicious(canvas, payload) {
    if (!payload.total_records) {
      emptyState(canvas, "No records analysed yet");
      return;
    }
    new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: ["Normal", "Malicious"],
        datasets: [
          {
            data: [payload.normal, payload.malicious],
            backgroundColor: ["#22c55e", "#f43f5e"],
            borderColor: "#0b1120",
            borderWidth: 3,
            hoverOffset: 8
          }
        ]
      },
      options: baseOptions({
        cutout: "62%",
        plugins: {
          legend: { position: "bottom", labels: { color: TEXT, boxWidth: 12 } },
          tooltip: {
            backgroundColor: "#0b1120",
            borderColor: "#22d3ee",
            borderWidth: 1,
            callbacks: {
              label: (context) => {
                const total = context.dataset.data.reduce((a, b) => a + b, 0) || 1;
                return ` ${context.label}: ${context.raw} (${((context.raw / total) * 100).toFixed(2)}%)`;
              }
            }
          }
        },
        scales: {}
      })
    });
  }

  function renderAttackTypes(canvas, payload) {
    const entries = toEntries(payload.attack_types).sort((a, b) => b.value - a.value);
    if (!entries.length) {
      emptyState(canvas, "No attack categories detected");
      return;
    }
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: entries.map((item) => item.label),
        datasets: [
          {
            label: "Records",
            data: entries.map((item) => item.value),
            backgroundColor: entries.map((_, index) => PALETTE[index % PALETTE.length]),
            borderRadius: 6
          }
        ]
      },
      options: baseOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: TEXT }, grid: { color: GRID }, beginAtZero: true },
          y: { ticks: { color: TEXT }, grid: { display: false } }
        }
      })
    });
  }

  function renderProtocols(canvas, payload) {
    const entries = toEntries(payload.protocols).sort((a, b) => b.value - a.value);
    if (!entries.length) {
      emptyState(canvas, "No protocol data available");
      return;
    }
    new Chart(canvas, {
      type: "polarArea",
      data: {
        labels: entries.map((item) => item.label),
        datasets: [
          {
            data: entries.map((item) => item.value),
            backgroundColor: entries.map((_, index) => PALETTE[index % PALETTE.length] + "cc"),
            borderWidth: 0
          }
        ]
      },
      options: baseOptions({
        plugins: { legend: { position: "right", labels: { color: TEXT, boxWidth: 12 } } },
        scales: {
          r: {
            ticks: { color: TEXT, backdropColor: "transparent" },
            grid: { color: GRID },
            angleLines: { color: GRID }
          }
        }
      })
    });
  }

  function renderDirection(canvas, payload) {
    const source = payload.directions || {};
    const labels = Object.keys(source);
    if (!labels.length) {
      emptyState(canvas, "No direction data available");
      return;
    }
    const colors = labels.map((label) => {
      if (label === "UNIDIRECTIONAL") return "#f59e0b";
      if (label === "FORWARD") return "#22d3ee";
      if (label === "REVERSE") return "#a78bfa";
      return "#64748b";
    });
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Flows",
            data: labels.map((label) => source[label]),
            backgroundColor: colors,
            borderRadius: 6
          }
        ]
      },
      options: baseOptions({
        plugins: { legend: { display: false } }
      })
    });
  }

  function renderProbability(canvas, payload) {
    const histogram = payload.probability_histogram || {};
    const labels = Object.keys(histogram);
    if (!labels.length) {
      emptyState(canvas, "Analyse a dataset to see probabilities");
      return;
    }
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Records",
            data: labels.map((key) => histogram[key]),
            backgroundColor: labels.map((label) => {
              const start = parseFloat(label.split("-")[0]);
              return start >= 0.5 ? "#f43f5e" : start >= 0.3 ? "#f59e0b" : "#22c55e";
            }),
            borderRadius: 4
          }
        ]
      },
      options: baseOptions({
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: TEXT, maxRotation: 60, minRotation: 30 }, grid: { display: false } },
          y: { ticks: { color: TEXT }, grid: { color: GRID }, beginAtZero: true }
        }
      })
    });
  }

  function renderModels(canvas, payload) {
    const models = (payload.models || []).filter((model) => model.model_name);
    if (!models.length) {
      emptyState(canvas, "Train the models to compare performance");
      return;
    }
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: models.map((model) => model.model_name),
        datasets: [
          { label: "Accuracy", data: models.map((m) => m.accuracy || 0), backgroundColor: "#22d3ee" },
          { label: "Precision", data: models.map((m) => m.precision || 0), backgroundColor: "#a78bfa" },
          { label: "Recall", data: models.map((m) => m.recall || 0), backgroundColor: "#22c55e" },
          { label: "F1", data: models.map((m) => m.f1_score || 0), backgroundColor: "#f59e0b" },
          { label: "ROC-AUC", data: models.map((m) => m.roc_auc || 0), backgroundColor: "#f43f5e" }
        ]
      },
      options: baseOptions({
        plugins: { legend: { labels: { color: TEXT, boxWidth: 12 } } },
        scales: {
          x: { ticks: { color: TEXT, maxRotation: 40, minRotation: 20 }, grid: { display: false } },
          y: { ticks: { color: TEXT }, grid: { color: GRID }, beginAtZero: true, suggestedMax: 1 }
        }
      })
    });
  }

  function renderCharts() {
    const payload = readPayload();
    if (!payload) {
      return;
    }
    const renderers = {
      chartNormalMalicious: renderNormalMalicious,
      chartAttackTypes: renderAttackTypes,
      chartProtocols: renderProtocols,
      chartDirection: renderDirection,
      chartProbability: renderProbability,
      chartModels: renderModels
    };
    (window.CYBER_CHARTS || Object.keys(renderers)).forEach(function (id) {
      const canvas = document.getElementById(id);
      if (canvas && renderers[id]) {
        renderers[id](canvas, payload);
      }
    });
  }

  window.initUploadDropZone = function () {
    const zone = document.getElementById("dropZone");
    const input = document.getElementById("datasetInput");
    const label = document.getElementById("fileName");
    if (!zone || !input || !label) {
      return;
    }
    zone.addEventListener("click", function () {
      input.click();
    });
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", function (event) {
      if (event.dataTransfer && event.dataTransfer.files.length) {
        input.files = event.dataTransfer.files;
        label.textContent = event.dataTransfer.files[0].name;
      }
    });
    input.addEventListener("change", function () {
      label.textContent = input.files.length ? input.files[0].name : "No file selected";
    });
  };

  window.initLiveSimulation = function () {
    const startBtn = document.getElementById("startBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const clearBtn = document.getElementById("clearBtn");
    const liveScanBtn = document.getElementById("liveScanBtn");
    const select = document.getElementById("uploadSelect");
    const intervalRange = document.getElementById("intervalRange");
    const intervalValue = document.getElementById("intervalValue");
    const streamBody = document.getElementById("streamBody");
    const stateLabel = document.getElementById("engineState");
    const dot = document.getElementById("engineDot");
    const statSeen = document.getElementById("statSeen");
    const statNormal = document.getElementById("statNormal");
    const statMalicious = document.getElementById("statMalicious");
    if (!startBtn || !streamBody) {
      return;
    }

    let timer = null;
    let liveMode = false;
    const counters = { seen: 0, normal: 0, malicious: 0 };

    function setState(state) {
      stateLabel.textContent = state;
      if (state === "alert") {
        dot.className = "pulse-dot danger";
      } else if (state === "scanning") {
        dot.className = "pulse-dot";
        dot.style.background = "#a78bfa";
      } else {
        dot.className = "pulse-dot";
        dot.style.background = "";
      }
    }

    function resetCounters() {
      counters.seen = 0;
      counters.normal = 0;
      counters.malicious = 0;
      statSeen.textContent = "0";
      statNormal.textContent = "0";
      statMalicious.textContent = "0";
    }

    function addRow(record) {
      const placeholder = document.getElementById("placeholder");
      if (placeholder) {
        placeholder.remove();
      }
      const row = document.createElement("tr");
      row.className = "live-row-enter";
      const malicious = record.prediction === "MALICIOUS";
      row.innerHTML =
        `<td class="mono">${record.source_ip}</td>` +
        `<td class="mono">${record.destination_ip}</td>` +
        `<td>${record.protocol}</td>` +
        `<td class="numeric">${record.packet_count}</td>` +
        `<td>${record.traffic_direction}</td>` +
        `<td><span class="badge-soft ${malicious ? "badge-malicious" : "badge-normal"}">${record.prediction}</span></td>` +
        `<td class="numeric">${Number(record.threat_probability).toFixed(2)}%</td>` +
        `<td>${malicious ? record.attack_type : "—"}</td>`;
      streamBody.prepend(row);
      while (streamBody.children.length > 60) {
        streamBody.removeChild(streamBody.lastChild);
      }

      counters.seen += 1;
      if (malicious) {
        counters.malicious += 1;
      } else {
        counters.normal += 1;
      }
      statSeen.textContent = counters.seen;
      statNormal.textContent = counters.normal;
      statMalicious.textContent = counters.malicious;
      setState(malicious ? "alert" : "monitoring");

      if (window.radarPush) {
        window.radarPush(record);
      }
    }

    async function tick() {
      let uploadId = select ? select.value : "";
      let endpoint = "";
      if (liveMode) {
        endpoint = "/api/simulation/live";
      } else if (uploadId) {
        endpoint = `/api/simulation/${uploadId}`;
      }
      if (!endpoint) {
        stop();
        setState("no dataset");
        return;
      }
      try {
        const response = await fetch(`${endpoint}?limit=1`);
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || "simulation endpoint failed");
        }
        const data = await response.json();
        if (!data.records || !data.records.length) {
          setState("no records");
          return;
        }
        addRow(data.records[0]);
      } catch (error) {
        setState("error");
      }
    }

    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      liveMode = false;
      startBtn.disabled = false;
      pauseBtn.disabled = true;
      if (liveScanBtn) {
        liveScanBtn.disabled = false;
        liveScanBtn.innerHTML = "&#128225; Start Live Scan";
      }
      if (stateLabel.textContent !== "error") {
        setState("idle");
      }
    }

    function start() {
      if (!select || !select.value) {
        return;
      }
      liveMode = false;
      stop();
      const delay = intervalRange ? parseInt(intervalRange.value, 10) : 1200;
      tick();
      timer = setInterval(tick, delay);
      startBtn.disabled = true;
      pauseBtn.disabled = false;
    }

    function startLiveScan() {
      liveMode = true;
      stop();
      const delay = intervalRange ? parseInt(intervalRange.value, 10) : 1200;
      tick();
      timer = setInterval(tick, delay);
      startBtn.disabled = true;
      pauseBtn.disabled = false;
      if (liveScanBtn) {
        liveScanBtn.disabled = true;
        liveScanBtn.innerHTML = "&#128225; Scanning...";
      }
      setState("scanning");
    }

    startBtn.addEventListener("click", start);
    pauseBtn.addEventListener("click", stop);
    if (liveScanBtn) {
      liveScanBtn.addEventListener("click", startLiveScan);
    }
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        stop();
        streamBody.innerHTML =
          '<tr id="placeholder"><td colspan="8" class="text-secondary">Press Start to begin the simulation.</td></tr>';
        resetCounters();
        if (window.resetRadar) {
          window.resetRadar();
        }
      });
    }
    if (intervalRange && intervalValue) {
      intervalRange.addEventListener("input", function () {
        intervalValue.textContent = intervalRange.value;
        if (timer) {
          start();
        }
      });
    }
  };

  document.addEventListener("DOMContentLoaded", function () {
    renderCharts();
    if (window.initUploadDropZone && document.getElementById("dropZone")) {
      initUploadDropZone();
    }
    if (window.initLiveSimulation && document.getElementById("streamBody")) {
      initLiveSimulation();
    }
    if (window.initRadar && document.getElementById("radarCanvas")) {
      initRadar();
    }
  });

  window.initRadar = function () {
    const canvas = document.getElementById("radarCanvas");
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = 420;
    const displayHeight = 420;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cx = displayWidth / 2;
    const cy = displayHeight / 2;
    const maxRadius = 190;
    const ringStep = maxRadius / 3;

    let sweepAngle = 0;
    const sweepSpeed = (2 * Math.PI) / 3.2;
    const blips = [];
    const maxBlips = 80;
    let threats = 0;
    let uniqueIps = new Set();
    let packetsThisSecond = 0;
    let lastSecond = Math.floor(Date.now() / 1000);

    function ipHash(ip) {
      let hash = 0;
      const text = String(ip || "0.0.0.0");
      for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash |= 0;
      }
      return Math.abs(hash);
    }

    function ipAngle(ip) {
      return ((ipHash(ip) % 360) * Math.PI) / 180;
    }

    function ipRadius(packetCount, malicious) {
      const base = Math.log10(Math.max(parseInt(packetCount, 10) || 1, 1)) / 4;
      const r = 30 + Math.min(base, 1) * (maxRadius - 35);
      return Math.max(20, Math.min(r, maxRadius - 10));
    }

    function pushBlip(record) {
      const malicious = record.prediction === "MALICIOUS";
      const angle = ipAngle(record.source_ip);
      const radius = ipRadius(record.packet_count, malicious);
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      blips.push({
        x,
        y,
        color: malicious ? "#f43f5e" : "#22d3ee",
        born: Date.now(),
        life: malicious ? 3500 : 2200,
        maxLife: malicious ? 3500 : 2200,
        radius: malicious ? 3.2 : 2.2,
        malicious,
      });
      packetsThisSecond += 1;
      uniqueIps.add(record.source_ip);
      if (malicious) {
        threats += 1;
      }
      while (blips.length > maxBlips) {
        blips.shift();
      }
    }

    function drawGrid() {
      ctx.strokeStyle = "rgba(148,163,184,0.14)";
      ctx.lineWidth = 1;
      for (let i = 1; i <= 3; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, ringStep * i, 0, Math.PI * 2);
        ctx.stroke();
      }
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(a) * maxRadius, cy + Math.sin(a) * maxRadius);
        ctx.stroke();
      }
    }

    function drawSweep() {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(sweepAngle);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(maxRadius, 0);
      ctx.strokeStyle = "rgba(34,211,238,0.75)";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();

      const grad = ctx.createConicGradient
        ? ctx.createConicGradient(sweepAngle, cx, cy)
        : ctx.createLinearGradient(cx, cy, cx + Math.cos(sweepAngle) * maxRadius, cy + Math.sin(sweepAngle) * maxRadius);
      if (grad.addColorStop) {
        grad.addColorStop(0, "rgba(34,211,238,0.28)");
        grad.addColorStop(0.12, "rgba(34,211,238,0.08)");
        grad.addColorStop(0.25, "rgba(34,211,238,0)");
        grad.addColorStop(1, "rgba(34,211,238,0)");
      }
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, maxRadius, sweepAngle - 0.55, sweepAngle, false);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.restore();
    }

    function drawBlips() {
      const now = Date.now();
      for (let i = blips.length - 1; i >= 0; i--) {
        const blip = blips[i];
        const age = now - blip.born;
        if (age > blip.life) {
          blips.splice(i, 1);
          continue;
        }
        const alpha = 1 - age / blip.life;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(blip.x, blip.y, blip.radius, 0, Math.PI * 2);
        ctx.fillStyle = blip.color;
        ctx.shadowColor = blip.color;
        ctx.shadowBlur = blip.malicious ? 14 : 8;
        ctx.fill();
        if (blip.malicious) {
          ctx.beginPath();
          ctx.arc(blip.x, blip.y, blip.radius + 4 + Math.sin(now / 120) * 2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(244,63,94,0.35)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    function updateStats() {
      const now = Math.floor(Date.now() / 1000);
      if (now !== lastSecond) {
        const pps = document.getElementById("radarPps");
        const threatsEl = document.getElementById("radarThreats");
        const coverageEl = document.getElementById("radarCoverage");
        if (pps) pps.textContent = packetsThisSecond;
        if (threatsEl) threatsEl.textContent = threats;
        if (coverageEl) coverageEl.textContent = Math.min(100, uniqueIps.size * 7) + "%";
        packetsThisSecond = 0;
        lastSecond = now;
      }
    }

    function frame() {
      ctx.clearRect(0, 0, displayWidth, displayHeight);
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, maxRadius, 0, Math.PI * 2);
      ctx.clip();
      drawGrid();
      drawSweep();
      drawBlips();
      ctx.restore();
      sweepAngle += sweepSpeed * 0.016;
      if (sweepAngle > Math.PI * 2) {
        sweepAngle -= Math.PI * 2;
      }
      updateStats();
      requestAnimationFrame(frame);
    }

    frame();

    window.radarPush = pushBlip;
    window.resetRadar = function () {
      blips.length = 0;
      threats = 0;
      uniqueIps = new Set();
      packetsThisSecond = 0;
    };
  };
})();