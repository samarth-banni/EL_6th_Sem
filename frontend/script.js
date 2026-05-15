const state = {
  videoFile: null,
  liveTimer: null,
  riskTimer: null,
  animationFrame: null,
  isAnalyzing: false,
  lastAnalysis: null,
  forecast: null,
  risk: null,
  alertSent: false,
  lastAlertAt: 0,
  displayBoxes: [],
  targetBoxes: [],
  location: "detecting",
  weather: "sunny",
};

const $ = (id) => document.getElementById(id);
const video = $("trafficVideo");
const overlay = $("overlayCanvas");
const overlayCtx = overlay.getContext("2d");
const capture = $("captureCanvas");
const captureCtx = capture.getContext("2d");

function setMessage(text) {
  $("message").textContent = text;
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function markApiOnline() {
  $("apiStatus").textContent = "API online";
  $("apiStatus").className = "pill ok";
}

function markApiOffline() {
  $("apiStatus").textContent = "API offline";
  $("apiStatus").className = "pill bad";
}

function renderModelStatus(health) {
  $("modelStatus").textContent = `YOLO ${health.detector_loaded ? "loaded" : "ready"} | TFT ${
    health.forecaster_loaded ? "on" : "fallback"
  }`;
  $("modelStatus").className = "pill ok";
}

async function refreshStatus() {
  try {
    const health = await requestJson("/health");
    markApiOnline();
    renderModelStatus(health);
  } catch {
    markApiOffline();
    $("modelStatus").textContent = "Models waiting";
    $("modelStatus").className = "pill neutral";
  }

  try {
    const telegram = await requestJson("/alerts/telegram/status");
    $("telegramStatus").textContent = telegram.configured ? "Telegram ready" : "Telegram not set";
    $("telegramStatus").className = telegram.configured ? "pill ok" : "pill bad";
  } catch {
    $("telegramStatus").textContent = "Telegram unknown";
    $("telegramStatus").className = "pill neutral";
  }
}

function setLiveStatus(text, mode = "neutral") {
  $("liveStatus").textContent = text;
  $("liveStatus").className = `pill ${mode}`;
}

function setLocationWeather(location, weather) {
  state.location = location || "unknown";
  state.weather = weather || "sunny";
  $("locationText").textContent = state.location;
  $("weatherText").textContent = titleCase(state.weather);
  $("liveWeather").textContent = titleCase(state.weather);
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(Math.round(rect.width * dpr), 1);
  overlay.height = Math.max(Math.round(rect.height * dpr), 1);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function videoDrawBox() {
  const rect = video.getBoundingClientRect();
  if (!video.videoWidth || !video.videoHeight) {
    return { left: 0, top: 0, width: rect.width, height: rect.height };
  }
  const videoRatio = video.videoWidth / video.videoHeight;
  const boxRatio = rect.width / rect.height;
  let width = rect.width;
  let height = rect.height;
  let left = 0;
  let top = 0;

  if (boxRatio > videoRatio) {
    width = rect.height * videoRatio;
    left = (rect.width - width) / 2;
  } else {
    height = rect.width / videoRatio;
    top = (rect.height - height) / 2;
  }
  return { left, top, width, height };
}

function animateOverlay() {
  smoothBoxes();
  drawDetections();
  state.animationFrame = requestAnimationFrame(animateOverlay);
}

function smoothBoxes() {
  const maxLength = Math.max(state.displayBoxes.length, state.targetBoxes.length);
  const next = [];
  for (let index = 0; index < maxLength; index += 1) {
    const current = state.displayBoxes[index];
    const target = state.targetBoxes[index];
    if (!target) continue;
    if (!current) {
      next.push({ ...target, alpha: 0.35 });
      continue;
    }
    next.push({
      ...target,
      x1: lerp(current.x1, target.x1, 0.22),
      y1: lerp(current.y1, target.y1, 0.22),
      x2: lerp(current.x2, target.x2, 0.22),
      y2: lerp(current.y2, target.y2, 0.22),
      confidence: lerp(current.confidence, target.confidence, 0.18),
      alpha: Math.min((current.alpha || 0.8) + 0.08, 1),
    });
  }
  state.displayBoxes = next;
}

function lerp(start, end, amount) {
  return start + (end - start) * amount;
}

function drawDetections() {
  const rect = video.getBoundingClientRect();
  overlayCtx.clearRect(0, 0, rect.width, rect.height);
  if (!video.videoWidth || !video.videoHeight) return;

  const drawBox = videoDrawBox();
  const scaleX = drawBox.width / video.videoWidth;
  const scaleY = drawBox.height / video.videoHeight;

  for (const box of state.displayBoxes) {
    const x = drawBox.left + box.x1 * scaleX;
    const y = drawBox.top + box.y1 * scaleY;
    const width = (box.x2 - box.x1) * scaleX;
    const height = (box.y2 - box.y1) * scaleY;
    const label = `${box.label} ${box.confidence.toFixed(2)}`;
    const color = colorForLabel(box.label);

    overlayCtx.globalAlpha = box.alpha || 1;
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeRect(x, y, width, height);

    overlayCtx.font = "700 13px Arial";
    const labelWidth = overlayCtx.measureText(label).width + 14;
    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(x, Math.max(y - 25, 0), labelWidth, 24);
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(label, x + 7, Math.max(y - 8, 17));
    overlayCtx.globalAlpha = 1;
  }
}

function colorForLabel(label) {
  return {
    car: "#2f6fed",
    bus: "#c9472c",
    truck: "#8a4fd3",
    motorcycle: "#117c6f",
  }[label] || "#17202a";
}

function captureFrameBase64() {
  if (!video.videoWidth || !video.videoHeight) return null;
  const maxWidth = 960;
  const ratio = Math.min(maxWidth / video.videoWidth, 1);
  capture.width = Math.round(video.videoWidth * ratio);
  capture.height = Math.round(video.videoHeight * ratio);
  captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
  return capture.toDataURL("image/jpeg", 0.78).split(",")[1];
}

async function analyzeCurrentFrame() {
  if (state.isAnalyzing || video.paused || video.ended) return;
  const frameBase64 = captureFrameBase64();
  if (!frameBase64) return;

  state.isAnalyzing = true;
  try {
    const result = await requestJson("/stream/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_base64: frameBase64,
        road_capacity: Number($("roadCapacity").value),
      }),
    });
    markApiOnline();
    renderModelStatus({ detector_loaded: true, forecaster_loaded: false });
    state.lastAnalysis = result;
    state.targetBoxes = result.boxes.slice().sort((a, b) => a.x1 - b.x1);
    renderLiveAnalysis(result);
    $("lastAnalyzed").textContent = new Date().toLocaleTimeString();
  } catch (error) {
    setMessage(`Live frame analysis failed: ${error.message}`);
  } finally {
    state.isAnalyzing = false;
  }
}

function renderLiveAnalysis(result) {
  const counts = result.counts || {};
  $("liveCount").textContent = `${result.total_vehicles} vehicles`;
  $("chaosIndex").textContent = formatNumber(result.chaos_index, 4);
  $("panelTotalVehicles").textContent = result.total_vehicles;
  $("panelChaosIndex").textContent = formatNumber(result.chaos_index, 4);
  $("panelCars").textContent = counts.car || 0;
  $("panelBikes").textContent = counts.motorcycle || 0;
  $("panelBuses").textContent = counts.bus || 0;
  $("panelTrucks").textContent = counts.truck || 0;

  $("objectList").innerHTML = result.boxes
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .map(
      (box) => `
        <div class="object-row">
          <span class="dot" style="background:${colorForLabel(box.label)}"></span>
          <strong>${box.label}</strong>
          <span>${box.confidence.toFixed(2)}</span>
        </div>
      `
    )
    .join("");
}

async function evaluateRisk(sendAlert = false) {
  if (!state.lastAnalysis) return;
  try {
    const now = Date.now();
    const allowAlert = sendAlert || (!state.alertSent && now - state.lastAlertAt > 120000);
    const result = await requestJson("/risk/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "Traffic Intelligence Dashboard",
        vehicle_count: state.lastAnalysis.total_vehicles,
        chaos_index: state.lastAnalysis.chaos_index,
        weather: state.weather,
        location: state.location,
        chaos_threshold: Number($("chaosThreshold").value),
        volume_threshold: Number($("volumeThreshold").value),
        send_alert: allowAlert,
      }),
    });

    state.risk = result;
    state.forecast = result.forecast;
    if (result.alert_sent) {
      state.alertSent = true;
      state.lastAlertAt = now;
    }
    renderRisk(result);
    renderForecast(result.forecast);
    setMessage(result.alert_sent ? result.alert_detail : "Live risk updated.");
  } catch (error) {
    setMessage(`Risk check failed: ${error.message}`);
  }
}

function renderRisk(result) {
  $("liveWeather").textContent = `${titleCase(result.weather)} | ${titleCase(result.time_period)}`;
  $("liveRisk").textContent = `Risk: ${titleCase(result.risk_level)}`;
  $("liveRisk").className = `risk-${result.risk_level}`;
  $("panelRisk").textContent = `Risk: ${titleCase(result.risk_level)}`;
  $("predictionLevel").textContent = titleCase(result.prediction_level);
  $("tftConfidence").textContent = `${formatNumber(result.tft_confidence, 1)}%`;
  $("vehicleDensity").textContent = titleCase(result.vehicle_density);
  $("riskReason").textContent = result.reason;
}

function renderForecast(result) {
  const predictions = result.predictions || [];
  const finalPoint = predictions[predictions.length - 1];
  $("modelSource").textContent = result.metadata?.model_source || "--";
  if (finalPoint) $("predictionLevel").title = `Predicted volume: ${formatNumber(finalPoint.predicted_volume, 0)}`;

  const importance = Object.entries(result.interpretation || {});
  const maxImportance = Math.max(...importance.map(([, value]) => value), 1);
  $("importanceList").innerHTML = importance
    .map(([name, value]) => {
      const width = Math.max((value / maxImportance) * 100, 2);
      return `
        <div class="importance-row">
          <span>${name}</span>
          <span class="track"><span class="fill" style="width:${width}%"></span></span>
          <strong>${formatNumber(value, 2)}</strong>
        </div>
      `;
    })
    .join("");
}

async function useSystemLocation() {
  if (!navigator.geolocation) {
    setLocationWeather("unknown", "sunny");
    setMessage("Location is not supported. Weather fallback is sunny.");
    return;
  }

  setMessage("Detecting location and weather...");
  navigator.geolocation.getCurrentPosition(
    async (position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      const location = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
      try {
        const weather = await fetchWeather(lat, lon);
        setLocationWeather(location, weather);
        setMessage(`Auto location and weather loaded: ${titleCase(weather)}.`);
        evaluateRisk(false);
      } catch (error) {
        setLocationWeather(location, "sunny");
        setMessage(`Location loaded. Weather API failed, using sunny fallback.`);
      }
    },
    () => {
      setLocationWeather("unknown", "sunny");
      setMessage("Location permission blocked. Enable browser location permission for automatic weather.");
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
}

async function fetchWeather(latitude, longitude) {
  const params = new URLSearchParams({
    latitude,
    longitude,
    current: "weather_code,precipitation,rain,showers,snowfall,cloud_cover",
    timezone: "auto",
  });
  const data = await requestJson(`https://api.open-meteo.com/v1/forecast?${params.toString()}`);
  const current = data.current || {};
  const code = Number(current.weather_code);
  const rain = Number(current.rain || 0) + Number(current.showers || 0) + Number(current.precipitation || 0);
  const cloud = Number(current.cloud_cover || 0);

  if (rain > 0 || (code >= 51 && code <= 67) || (code >= 80 && code <= 82)) return "rain";
  if (code === 45 || code === 48) return "fog";
  if (cloud >= 55 || (code >= 2 && code <= 3)) return "cloudy";
  return "sunny";
}

function titleCase(value) {
  const text = String(value || "");
  return text.slice(0, 1).toUpperCase() + text.slice(1);
}

function startLiveAnalysis() {
  if (!state.videoFile) {
    setMessage("Choose a video first.");
    return;
  }
  if (state.liveTimer) return;

  const interval = 1000;
  state.alertSent = false;
  state.liveTimer = setInterval(analyzeCurrentFrame, interval);
  state.riskTimer = setInterval(() => evaluateRisk(false), 12000);
  setLiveStatus("Live analyzing", "ok");
  setMessage("Live analysis started. Smooth boxes are drawn on the video.");
  if (video.paused) video.play();
  analyzeCurrentFrame();
  setTimeout(() => evaluateRisk(false), 2500);
}

function stopLiveAnalysis() {
  clearInterval(state.liveTimer);
  clearInterval(state.riskTimer);
  state.liveTimer = null;
  state.riskTimer = null;
  setLiveStatus("Live stopped", "neutral");
  setMessage("Live analysis stopped.");
}

$("videoInput").addEventListener("change", (event) => {
  state.videoFile = event.target.files[0] || null;
  stopLiveAnalysis();
  state.lastAnalysis = null;
  state.forecast = null;
  state.alertSent = false;
  state.displayBoxes = [];
  state.targetBoxes = [];
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!state.videoFile) {
    $("fileLabel").textContent = "Choose traffic video";
    return;
  }

  $("fileLabel").textContent = state.videoFile.name;
  video.src = URL.createObjectURL(state.videoFile);
  video.load();
  setMessage("Video loaded. Press Start Live Analysis.");
});

$("startLiveBtn").addEventListener("click", startLiveAnalysis);
$("locationBtn").addEventListener("click", useSystemLocation);
$("stopLiveBtn").addEventListener("click", stopLiveAnalysis);
$("alertBtn").addEventListener("click", () => evaluateRisk(true));
video.addEventListener("loadedmetadata", resizeOverlay);
video.addEventListener("play", resizeOverlay);
video.addEventListener("pause", () => setLiveStatus(state.liveTimer ? "Live paused" : "Live idle", "neutral"));
video.addEventListener("ended", stopLiveAnalysis);
window.addEventListener("resize", resizeOverlay);

refreshStatus();
setInterval(refreshStatus, 10000);
resizeOverlay();
animateOverlay();
useSystemLocation();
