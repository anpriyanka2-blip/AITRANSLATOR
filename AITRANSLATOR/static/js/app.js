const SignBridge = (() => {
  const byId = (id) => document.getElementById(id);
  const routes = window.APP_ROUTES || {};
  let selectedVoice = null;

  function loadVoice() {
    const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    selectedVoice = voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith("en")) || voices[0] || null;
  }

  function speak(text, lang = "en-IN") {
    const status = byId("speech-status");
    if (!("speechSynthesis" in window)) {
      if (status) status.textContent = "Speech is not supported in this browser.";
      return;
    }
    const cleanText = (text || "").trim();
    if (!cleanText || cleanText === "-") {
      if (status) status.textContent = "No speech text available.";
      return;
    }
    loadVoice();
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = lang;
    utterance.rate = 0.92;
    utterance.pitch = 1;
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.onstart = () => { if (status) status.textContent = `Speaking: ${cleanText}`; };
    utterance.onend = () => { if (status) status.textContent = "Speech output complete."; };
    utterance.onerror = () => { if (status) status.textContent = "Speech output was blocked. Click Speak again."; };
    window.speechSynthesis.speak(utterance);
  }

  if ("speechSynthesis" in window) {
    loadVoice();
    window.speechSynthesis.onvoiceschanged = loadVoice;
  }

  function updateLive(data) {
    const liveLabel = byId("live-label");
    const liveConfidence = byId("live-confidence");
    const stableLabel = byId("stable-label");
    const liveHindi = byId("live-hindi");
    const liveTelugu = byId("live-telugu");
    const liveKannada = byId("live-kannada");
    const startButton = byId("start-prediction");
    const stopButton = byId("stop-prediction");
    if (liveLabel) liveLabel.textContent = data.label || "Waiting";
    if (liveConfidence) liveConfidence.textContent = data.confidence || "0.0%";
    if (stableLabel) stableLabel.textContent = data.stable_label || "-";
    if (liveHindi) liveHindi.textContent = data.translations?.hindi || "-";
    if (liveTelugu) liveTelugu.textContent = data.translations?.telugu || "-";
    if (liveKannada) liveKannada.textContent = data.translations?.kannada || "-";
    if (startButton) startButton.disabled = data.active;
    if (stopButton) stopButton.disabled = !data.active;
  }

  async function pollPrediction() {
    if (!routes.current) return;
    try {
      const response = await fetch(routes.current, { cache: "no-store" });
      updateLive(await response.json());
    } catch {}
  }

  async function postLive(url) {
    const response = await fetch(url, { method: "POST", cache: "no-store" });
    const data = await response.json();
    updateLive(data);
    if (!data.active && data.label && data.label !== "No stable sign captured") speak(data.label);
  }

  function initLive() {
    const startButton = byId("start-prediction");
    const stopButton = byId("stop-prediction");
    const speakButton = byId("speak-live");
    if (startButton) startButton.addEventListener("click", () => postLive(routes.start));
    if (stopButton) stopButton.addEventListener("click", () => postLive(routes.stop));
    if (speakButton) speakButton.addEventListener("click", () => speak(byId("live-label")?.textContent || byId("stable-label")?.textContent));
    pollPrediction();
    if (startButton || stopButton) setInterval(pollPrediction, 700);
  }

  function initUpload() {
    const form = byId("upload-form");
    if (!form) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const result = byId("upload-result");
      result.innerHTML = '<div class="muted">Analyzing upload...</div>';
      const formData = new FormData(form);
      formData.append("language", byId("language")?.value || "english");
      const response = await fetch(routes.upload, { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        result.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        return;
      }
      result.innerHTML = `
        <div class="output-box">
          <span class="label-sm">${data.source_type} prediction</span>
          <strong>${data.label} - ${data.confidence}</strong>
          <div class="muted small mt-2">Hindi: ${data.translations.hindi} | Telugu: ${data.translations.telugu} | Kannada: ${data.translations.kannada}</div>
          ${data.output_media_url && data.output_media_type === "video" ? `<video class="w-100 mt-3 rounded border" src="${data.output_media_url}" controls autoplay muted loop></video>` : ""}
          ${data.output_media_url && data.output_media_type === "image" ? `<img class="img-fluid rounded mt-3 border" src="${data.output_media_url}" alt="${data.label} sign">` : ""}
          <div class="small muted mt-2">Uploaded ${data.source_type}: <a href="${data.media_url}" target="_blank">open file</a></div>
        </div>`;
      speak(data.label);
    });
  }

  function initConvert() {
    const convertButton = byId("convert-text");
    if (!convertButton) return;
    convertButton.addEventListener("click", async () => {
      const result = byId("text-result");
      result.innerHTML = '<div class="muted">Searching sign videos and frames...</div>';
      const response = await fetch(routes.text, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: byId("text-input").value, language: byId("language")?.value || "english" })
      });
      const data = await response.json();
      if (!response.ok) {
        result.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
        return;
      }
      result.innerHTML = data.matches.map((match) => `
        <div class="output-box mb-3">
          <span class="label-sm">${match.label}</span>
          <strong>${match.translation.hindi} | ${match.translation.telugu} | ${match.translation.kannada}</strong>
          ${match.video ? `<video class="w-100 mt-3 rounded border" src="${match.video}" controls autoplay muted loop></video>` : ""}
          ${!match.video ? `<div class="frames-strip mt-3">${match.frames.map((frame) => `<img src="${frame.url}" alt="${match.label} frame">`).join("")}</div>` : ""}
        </div>`).join("");
      speak(data.matches.map((match) => match.label).join(", "));
    });
  }

  function initGallery() {
    const search = byId("gallery-search");
    if (!search) return;
    search.addEventListener("input", (event) => {
      const query = event.target.value.toLowerCase();
      document.querySelectorAll(".sign-card").forEach((card) => {
        card.style.display = card.dataset.label.includes(query) ? "" : "none";
      });
    });
  }

  function initCharts() {
    if (!window.Chart || !window.CHART_DATA || !byId("accuracyChart")) return;
    const charts = window.CHART_DATA;
    new Chart(byId("accuracyChart"), {
      type: "line",
      data: {
        labels: charts.epochs,
        datasets: [
          { label: "Accuracy", data: charts.accuracy, borderColor: "#0f9f6e", backgroundColor: "rgba(15,159,110,.14)", tension: .35, fill: true },
          { label: "Loss", data: charts.loss, borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,.1)", tension: .35, fill: true }
        ]
      }
    });
    if (byId("classChart")) {
      new Chart(byId("classChart"), {
        type: "bar",
        data: { labels: charts.labels, datasets: [{ label: "Frames per class", data: charts.class_counts, backgroundColor: "#2563eb" }] }
      });
    }
  }

  function init() {
    initLive();
    initUpload();
    initConvert();
    initGallery();
    initCharts();
  }

  return { init, speak };
})();

document.addEventListener("DOMContentLoaded", SignBridge.init);
