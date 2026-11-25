"""Presentation helpers for the Streamlit interface."""

from base64 import b64encode
from hashlib import sha256
from html import escape
from io import BytesIO

import numpy as np
import soundfile as sf
import streamlit as st


STYLE = """
<style>
  :root { color-scheme: dark; }
  .stApp { background: #160f18; color: #f6edf4; }
  [data-testid="stHeader"], [data-testid="stDecoration"], footer { display: none; }
  [data-testid="stAppViewBlockContainer"] { max-width: none; padding: 0; }
  [data-testid="stMainBlockContainer"] { max-width: none; padding: 0; }
  .st-key-library_panel, .st-key-song_panel, .st-key-results_panel { min-height: 100vh; }
  .st-key-library_panel, .st-key-song_panel { padding: 1.5rem; }
  .st-key-results_panel { padding: 1.5rem 2rem; position: relative; }
  .st-key-library_panel { background: #1b131d; border-right: 1px solid #483047; }
  .st-key-song_panel { background: #1a121c; border-right: 1px solid #483047; }
  .st-key-results_panel { background: #160f18; }
  .eyebrow { color: #c6b3c5; font-size: .95rem; font-weight: 800; letter-spacing: .12em; }
  .panel-title { color: #f6edf4; font-size: 1.65rem; font-weight: 750; line-height: 1.15; margin: .2rem 0 .45rem; }
  .st-key-song_panel .panel-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .map-note { color: #c9b6c8; font-size: .95rem; position: absolute; right: 2rem; top: 1.5rem; }
  .map-placeholder {
    border: 1.5px dashed #4d334d;
    border-radius: 1.45rem;
    min-height: calc(100vh - 8.5rem);
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    text-align: center; padding: 2rem; box-sizing: border-box;
  }
  .map-placeholder h3 { font-size: 1.35rem; margin: 1.3rem 0 .7rem; color: #eff0f4; }
  .map-placeholder p { max-width: 27rem; color: #c6b3c5; font-size: 1rem; line-height: 1.35; }
  .orbit { position: relative; width: 8rem; height: 8rem; }
  .orbit::before, .orbit::after, .orbit span {
    content: ""; position: absolute; border-radius: 50%; inset: .2rem;
  }
  .orbit::before { border: 2px dashed #4d334d; }
  .orbit::after { inset: 1.45rem; border: 2px dashed #4d334d; }
  .orbit span { inset: 2.8rem; border: 2px solid #f461af; }
  .st-key-library_filter { margin: 1.1rem 0 1.2rem; }
  .st-key-library_filter input {
    background: #160f18; border: 1px solid #654665;
    border-radius: .75rem; min-height: 2.8rem; font-size: .95rem;
  }
  [class*="st-key-song_row_"], .st-key-selected_song_row {
    padding: .55rem .7rem; border-radius: .85rem; margin: .25rem 0;
  }
  .st-key-selected_song_row { background: #43233e; border-left: 3px solid #f461af; }
  [class*="st-key-song_row_"] [data-testid="stButton"] button,
  .st-key-selected_song_row [data-testid="stButton"] button {
    border: 0; background: transparent; text-align: left;
    justify-content: flex-start; font-size: 1.05rem; font-weight: 750;
  }
  [class*="st-key-song_row_"] [data-testid="stButton"] button:hover,
  .st-key-selected_song_row [data-testid="stButton"] button:hover { background: #543049; }
  .st-key-mix_player {
    background: #251827; border-radius: 1rem;
    padding: .5rem 1rem; margin-bottom: .7rem;
  }
  .wave-player {
    display: flex; align-items: center; gap: .75rem;
    width: 100%; min-height: 3.25rem; color: #f6edf4;
  }
  .wave-player audio { display: none; }
  .wave-toggle {
    display: flex; align-items: center; justify-content: center;
    flex: 0 0 2.8rem; width: 2.8rem; height: 2.8rem;
    border: 0; border-radius: 50%; background: #f461af; color: #1a1119;
    padding: 0; line-height: 1; font-size: 1.15rem; font-weight: 750; cursor: pointer;
  }
  .wave-toggle:hover { background: #ff7dbd; }
  .wave-seek {
    position: relative; flex: 1 1 auto; min-width: 0; height: 2.25rem;
    cursor: pointer; touch-action: none;
  }
  .wave-seek:focus-visible, .wave-toggle:focus-visible {
    outline: 2px solid #ff9dce; outline-offset: 3px;
  }
  .wave-bars {
    display: flex; align-items: center; gap: 2px;
    position: absolute; inset: 0; overflow: hidden;
  }
  .wave-bars span {
    display: block; flex: 1 0 2px; border-radius: 2px;
    background: #71556f;
  }
  .wave-bars-played {
    clip-path: inset(0 calc(100% - var(--progress)) 0 0);
    pointer-events: none;
  }
  .wave-bars-played span { background: #f461af; }
  .wave-time {
    flex: 0 0 auto; color: #c6b3c5; font-size: .9rem;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .wave-player-compact { gap: .4rem; min-height: 2.25rem; }
  .wave-player-compact .wave-toggle {
    flex-basis: 2.15rem; width: 2.15rem; height: 2.15rem;
    background: #271a29; border: 1px solid #654665;
    color: #f6edf4; font-size: .85rem;
  }
  .wave-player-compact .wave-toggle:hover { background: #543049; }
  .wave-player-compact .wave-seek { height: 2rem; }
  .separated-status {
    background: #251827; border: 1px solid #5b3d5a; border-radius: .7rem;
    color: #cbbbc9; padding: .65rem .4rem; text-align: center;
    font-weight: 700; margin-bottom: .75rem;
  }
  .stems-heading { margin: .4rem 0 .65rem; }
  .st-key-stem_row_vocals, .st-key-stem_row_drums, .st-key-stem_row_bass,
  .st-key-stem_row_guitar, .st-key-stem_row_piano, .st-key-stem_row_other,
  .st-key-selected_stem_row {
    border-radius: .8rem; padding: .18rem .48rem; margin: .05rem 0;
  }
  .st-key-selected_stem_row { background: #43233e; border-left: 3px solid #f461af; }
  .st-key-stem_row_vocals button, .st-key-stem_row_drums button,
  .st-key-stem_row_bass button, .st-key-stem_row_guitar button,
  .st-key-stem_row_piano button, .st-key-stem_row_other button,
  .st-key-selected_stem_row button { min-height: 2rem; padding: .15rem .2rem; font-size: .82rem; }
  .library-spacer { height: max(1rem, calc(100vh - 20rem)); }
  .st-key-library_drop {
    border: 1.5px dashed #5b3b59; border-radius: 1rem; padding: .85rem;
  }
  .st-key-library_drop [data-testid="stFileUploaderDropzone"] {
    background: transparent; border: 0; padding: .25rem;
  }
  .st-key-library_drop [data-testid="stFileUploaderDropzone"] button {
    background: #f461af; color: #1a1119; border: 0; border-radius: .75rem;
    font-size: 1.1rem; font-weight: 800; width: 100%; min-height: 3.2rem;
  }
  .st-key-library_drop [data-testid="stFileUploaderDropzone"] button:hover {
    background: #ff7dbd; color: #1a1119;
  }
  .st-key-library_drop [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
  .upload-hint {
    color: #c6b3c5; font-size: .92rem; line-height: 1.25;
    text-align: center; margin: .35rem 0 .1rem;
  }
  .st-key-play_all_action { margin-top: 1.2rem; }
  .st-key-play_all_action button {
    background: #251827; border: 1px solid #654665; color: #f6edf4;
    min-height: 3.15rem; font-size: 1rem; font-weight: 700;
  }
  .st-key-search_action {
    border: 1.5px solid #f461af; border-radius: .85rem;
    background: #251827; padding: .75rem 1rem; margin-top: 1.5rem;
  }
  .selected-label { color: #c6b3c5; font-size: .9rem; }
  .selected-name { color: #f6edf4; font-weight: 800; font-size: 1.2rem; }
  .st-key-search_action button {
    background: #f461af; color: #1a1119; border: 0; border-radius: .75rem;
    font-weight: 800; min-height: 3.1rem;
  }
  .st-key-search_action button:hover { background: #ff7dbd; color: #1a1119; }
  .st-key-result_card_1, .st-key-result_card_2, .st-key-result_card_3 {
    background: #251827; border: 1px solid #4d334d; border-radius: 1rem;
    padding: 1rem 1.25rem; margin-bottom: 1rem;
  }
  @media (max-width: 1000px) {
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) > [data-testid="stColumn"] {
      width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }
    .st-key-library_panel, .st-key-song_panel, .st-key-results_panel { min-height: auto; padding: 1.4rem; }
    .library-spacer { height: 1rem !important; }
    .map-placeholder { min-height: 28rem; }
  }
</style>
"""


EMPTY_MAP = """
<div class="map-placeholder">
  <div class="orbit" aria-hidden="true"><span></span></div>
  <h3>Select a stem, then run a search</h3>
  <p>Sounds similar to your chosen stem will appear here as a map.</p>
</div>
"""


def apply_style() -> None:
    st.html(STYLE)


def _waveform_heights(data: bytes, bars: int) -> tuple[np.ndarray, float]:
    """Measure peak amplitude in evenly spaced windows of the audio."""
    with sf.SoundFile(BytesIO(data)) as audio:
        duration = audio.frames / audio.samplerate
        edges = np.linspace(0, audio.frames, bars + 1, dtype=int)
        peaks = np.zeros(bars, dtype=float)
        for index in range(bars):
            audio.seek(int(edges[index]))
            samples = audio.read(int(edges[index + 1] - edges[index]), dtype="float32")
            if samples.size:
                peaks[index] = float(np.max(np.abs(samples)))
    heights = 5 + 29 * peaks / (float(peaks.max()) + 1e-8)
    return heights, duration


def waveform_player_html(
    data: bytes,
    *,
    key: str,
    label: str,
    mime_type: str = "audio/wav",
    bars: int = 36,
    compact: bool = False,
) -> str:
    """Build one audio control whose waveform is the playback progress and seek bar."""
    if mime_type not in {"audio/wav", "audio/mpeg", "audio/flac"}:
        raise ValueError("Unsupported audio type for waveform playback.")
    heights, duration = _waveform_heights(data, bars)
    bars_html = "".join(f'<span style="height:{height:.1f}px"></span>' for height in heights)
    player_id = f"wave-player-{sha256(key.encode()).hexdigest()[:16]}"
    source = f"data:{mime_type};base64,{b64encode(data).decode('ascii')}"
    safe_label = escape(label, quote=True)
    rounded_duration = round(duration)
    total = f"{rounded_duration // 60}:{rounded_duration % 60:02d}"
    time_html = "" if compact else f'<span class="wave-time">0:00 / {total}</span>'
    player_class = "wave-player wave-player-compact" if compact else "wave-player"
    html = f"""
<div id="{player_id}" class="{player_class}">
  <audio preload="metadata" src="{source}"></audio>
  <button class="wave-toggle" type="button" aria-label="Play {safe_label}"
          data-label="{safe_label}">▶</button>
  <div class="wave-seek" role="slider" tabindex="0" aria-label="Seek in {safe_label}"
       aria-valuemin="0" aria-valuemax="{int(round(duration))}" aria-valuenow="0"
       style="--progress: 0%" title="Click or drag to seek">
    <div class="wave-bars" aria-hidden="true">{bars_html}</div>
    <div class="wave-bars wave-bars-played" aria-hidden="true">{bars_html}</div>
  </div>
  {time_html}
</div>
<script>
(() => {{
  const root = document.getElementById("{player_id}");
  if (!root) return;
  const audio = root.querySelector("audio");
  const toggle = root.querySelector(".wave-toggle");
  const seek = root.querySelector(".wave-seek");
  const clock = root.querySelector(".wave-time");
  const fallbackDuration = {duration:.6f};
  const label = toggle.dataset.label;
  let frame = 0;
  const length = () => Number.isFinite(audio.duration) ? audio.duration : fallbackDuration;
  const format = seconds => {{
    const value = Math.max(0, Math.floor(seconds));
    return `${{Math.floor(value / 60)}}:${{String(value % 60).padStart(2, "0")}}`;
  }};
  const update = () => {{
    const duration = length();
    const current = Math.min(audio.currentTime || 0, duration);
    seek.style.setProperty("--progress", `${{duration ? current / duration * 100 : 0}}%`);
    seek.setAttribute("aria-valuemax", String(Math.round(duration)));
    seek.setAttribute("aria-valuenow", String(Math.round(current)));
    seek.setAttribute("aria-valuetext", `${{format(current)}} of ${{format(Math.round(duration))}}`);
    if (clock) clock.textContent = `${{format(current)}} / ${{format(Math.round(duration))}}`;
  }};
  const tick = () => {{
    frame = 0;
    update();
    if (!audio.paused) frame = requestAnimationFrame(tick);
  }};
  const setPlaying = playing => {{
    toggle.textContent = playing ? "❚❚" : "▶";
    toggle.setAttribute("aria-label", `${{playing ? "Pause" : "Play"}} ${{label}}`);
    cancelAnimationFrame(frame);
    frame = 0;
    update();
    if (playing) frame = requestAnimationFrame(tick);
  }};
  toggle.addEventListener("click", () => {{
    if (audio.paused) {{
      if (audio.currentTime >= length()) audio.currentTime = 0;
      audio.play();
    }} else audio.pause();
  }});
  audio.addEventListener("play", () => {{
    document.querySelectorAll(".wave-player audio").forEach(other => {{
      if (other !== audio) other.pause();
    }});
    setPlaying(true);
  }});
  audio.addEventListener("pause", () => setPlaying(false));
  audio.addEventListener("ended", () => setPlaying(false));
  audio.addEventListener("loadedmetadata", update);
  audio.addEventListener("timeupdate", update);
  const seekTo = event => {{
    const bounds = seek.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    audio.currentTime = fraction * length();
    update();
  }};
  seek.addEventListener("pointerdown", event => {{ seek.setPointerCapture(event.pointerId); seekTo(event); }});
  seek.addEventListener("pointermove", event => {{ if (seek.hasPointerCapture(event.pointerId)) seekTo(event); }});
  seek.addEventListener("pointerup", event => {{ if (seek.hasPointerCapture(event.pointerId)) seek.releasePointerCapture(event.pointerId); }});
  seek.addEventListener("keydown", event => {{
    let next = audio.currentTime;
    if (event.key === "ArrowRight") next += 1;
    else if (event.key === "ArrowLeft") next -= 1;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = length();
    else return;
    event.preventDefault();
    audio.currentTime = Math.max(0, Math.min(length(), next));
    update();
  }});
  update();
}})();
</script>
"""
    return html
