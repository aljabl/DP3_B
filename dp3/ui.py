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
  .st-key-library_panel {
    background: #1b131d; border-right: 1px solid #483047;
    min-height: calc(100dvh - 1rem);
  }
  .st-key-song_panel { background: #1a121c; border-right: 1px solid #483047; }
  .st-key-results_panel { background: #160f18; }
  .eyebrow { color: #c6b3c5; font-size: .95rem; font-weight: 800; letter-spacing: .12em; }
  .panel-title { color: #f6edf4; font-size: 1.65rem; font-weight: 750; line-height: 1.15; margin: .2rem 0 .45rem; }
  .artist-heading { color: #c6b3c5; font-size: 1rem; font-weight: 600; margin: 0 0 .8rem; }
  .library-artist { font-size: .85rem; margin: -.2rem 0 .35rem; padding-left: .45rem; }
  .result-title { color: #f6edf4; font-size: 1.25rem; font-weight: 700; margin: 0 0 .2rem; }
  .st-key-song_panel .panel-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stem-resize-handle {
    position: fixed; z-index: 100; width: 12px;
    cursor: col-resize; touch-action: none;
  }
  .stem-resize-handle::after {
    content: ""; position: absolute; left: 5px; top: 0; bottom: 0;
    width: 2px; background: #654665;
  }
  .stem-resize-handle:hover::after, .stem-resize-handle:focus-visible::after,
  .stem-resize-handle.is-dragging::after { background: #f461af; }
  .stem-resize-handle:focus-visible { outline: 2px solid #ff9dce; outline-offset: 2px; }
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
  .st-key-save_song_action {
    background: #251827; border: 1px solid #5b3d5a;
    border-radius: .85rem; padding: .75rem; margin: .8rem 0;
  }
  .st-key-save_song_action [data-testid="stButton"] button {
    background: #43233e; border: 1px solid #654665;
    color: #f6edf4; font-weight: 750;
  }
  .st-key-save_song_action [data-testid="stButton"] button:hover {
    background: #543049; border-color: #f461af; color: #f6edf4;
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
  .st-key-library_panel > [data-testid="stLayoutWrapper"]:has(.st-key-library_song_list) {
    flex: 1 1 0; min-height: 0; overflow-y: auto;
    overscroll-behavior: contain; scrollbar-width: thin;
  }
  .st-key-library_panel > [data-testid="stLayoutWrapper"]:has(.st-key-library_drop) {
    flex: 0 0 auto;
  }
  .st-key-library_drop {
    border: 1.5px dashed #5b3b59; border-radius: 1rem; padding: .85rem;
    max-height: 40vh; overflow-y: auto;
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
  .st-key-library_drop button[aria-label^="Remove "] {
    flex: 0 0 1.25rem; width: 1.25rem; height: 1.25rem; min-height: 0;
    padding: 0; border: 0; border-radius: 50%; background: transparent;
    color: #c6b3c5; display: inline-flex; align-items: center; justify-content: center;
  }
  .st-key-library_drop button[aria-label^="Remove "]:hover {
    background: #43233e; color: #f6edf4;
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
  .st-key-search_action [data-testid="stCheckbox"] label {
    align-items: center !important;
  }
  .st-key-search_action button[aria-label="Help for Search all stem types"] {
    flex: 0 0 1.25rem; width: 1.25rem; height: 1.25rem; min-height: 0;
    padding: 0; border: 0; border-radius: 50%; background: transparent;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .st-key-search_action [data-testid="stButton"] button {
    background: #f461af; color: #1a1119; border: 0; border-radius: .75rem;
    font-weight: 800; min-height: 3.1rem;
  }
  .st-key-search_action [data-testid="stButton"] button:hover {
    background: #ff7dbd; color: #1a1119;
  }
  .st-key-result_card_1, .st-key-result_card_2, .st-key-result_card_3 {
    background: #251827; border: 1px solid #4d334d; border-radius: 1rem;
    padding: 1rem 1.25rem; margin-bottom: 1rem;
  }
  @media (min-width: 1001px) {
    [data-testid="stMain"] { overflow: hidden; }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0; }
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) {
      height: 100vh; overflow: hidden;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) > [data-testid="stColumn"] {
      height: 100vh; min-height: 0; overflow: hidden;
    }
    .st-key-library_panel, .st-key-song_panel, .st-key-results_panel {
      min-height: 100vh !important; max-height: 100vh !important;
      overflow-y: auto;
    }
    .st-key-library_panel { overflow: hidden; }
    .map-placeholder { min-height: calc(100vh - 10rem); }
  }
  @media (max-width: 1000px) {
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"]:has(.st-key-library_panel) > [data-testid="stColumn"] {
      width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }
    .st-key-library_panel, .st-key-song_panel, .st-key-results_panel {
      min-height: auto; padding: 1.4rem;
    }
    .st-key-library_panel {
      min-height: 100dvh !important; max-height: 100dvh !important;
      overflow: hidden;
    }
    .map-placeholder { min-height: 28rem; }
    .stem-resize-handle { height: 12px; cursor: row-resize; }
    .stem-resize-handle::after {
      left: 0; right: 0; top: 5px; bottom: auto; width: auto; height: 2px;
    }
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


STEM_PANE_RESIZER = """
<script>
(() => {
  window.__stemPaneResizer?.destroy();
  const panel = document.querySelector('.st-key-song_panel');
  const song = panel?.closest('[data-testid="stColumn"]');
  const row = song?.parentElement;
  if (!row || row.getAttribute('data-testid') !== 'stHorizontalBlock') return;
  const columns = Array.from(row.children).filter(
    element => element.getAttribute('data-testid') === 'stColumn'
  );
  const index = columns.indexOf(song);
  const library = columns[0];
  const results = columns[index + 1];
  if (!results) return;

  const widthKey = 'stem-pane-width';
  const heightKey = 'stem-pane-height';
  if (__OPEN_PANE__) sessionStorage.removeItem(widthKey);
  const snapWidth = 280;
  const libraryFraction = 1.05 / 5.35;
  const songFraction = 1.3 / 5.35;
  const storedWidth = sessionStorage.getItem(widthKey);
  const savedWidth = Number(storedWidth);
  let width = storedWidth !== null && Number.isFinite(savedWidth) && savedWidth >= 0
    ? savedWidth : null;
  const storedHeight = sessionStorage.getItem(heightKey);
  const savedHeight = Number(storedHeight);
  let height = storedHeight !== null && Number.isFinite(savedHeight) && savedHeight >= 0
    ? savedHeight : null;
  const mobile = window.matchMedia('(max-width: 1000px)');
  const handle = document.createElement('div');
  handle.className = 'stem-resize-handle';
  handle.tabIndex = 0;
  handle.setAttribute('role', 'separator');
  handle.setAttribute('aria-label', 'Resize stem panel');
  handle.setAttribute('aria-orientation', mobile.matches ? 'horizontal' : 'vertical');
  handle.setAttribute('aria-valuemin', '0');
  handle.title = 'Drag to resize or hide the stem panel';
  document.body.appendChild(handle);

  const boundedWidth = value => Math.max(0, Math.min(
    value, row.getBoundingClientRect().width - library.getBoundingClientRect().width - 200
  ));
  const defaultWidth = () => boundedWidth(Math.max(
    snapWidth + 1, row.getBoundingClientRect().width * songFraction
  ));
  function position() {
    const bounds = song.getBoundingClientRect();
    if (mobile.matches) {
      handle.style.position = 'fixed';
      handle.style.left = `${bounds.left}px`;
      handle.style.top = `${bounds.bottom - 6}px`;
      handle.style.width = `${bounds.width}px`;
      handle.style.bottom = 'auto';
      handle.style.height = '12px';
    } else {
      handle.style.position = 'fixed';
      handle.style.left = `${bounds.right - 6}px`;
      handle.style.top = `${Math.max(0, bounds.top)}px`;
      handle.style.width = '12px';
      handle.style.bottom = '0';
      handle.style.height = 'auto';
    }
  }
  function render() {
    handle.setAttribute('aria-orientation', mobile.matches ? 'horizontal' : 'vertical');
    if (mobile.matches) {
      library.style.removeProperty('width');
      library.style.removeProperty('flex');
      song.style.removeProperty('width');
      song.style.removeProperty('flex');
      results.style.removeProperty('width');
      results.style.removeProperty('flex');
      song.style.setProperty('max-height', height === null ? 'none' : `${height}px`);
      song.style.setProperty('overflow', height === null ? 'visible' : 'auto');
      handle.setAttribute('aria-valuemax', String(Math.round(song.scrollHeight)));
      handle.setAttribute('aria-valuenow', String(Math.round(height ?? song.scrollHeight)));
    } else {
      const libraryPercent = `${libraryFraction * 100}%`;
      library.style.setProperty('flex', `0 0 ${libraryPercent}`, 'important');
      library.style.setProperty('width', libraryPercent, 'important');
      song.style.removeProperty('max-height');
      song.style.setProperty('overflow', 'hidden');
      if (width === null) width = defaultWidth();
      width = boundedWidth(width);
      song.style.setProperty('flex', `0 0 ${width}px`, 'important');
      song.style.setProperty('width', `${width}px`, 'important');
      song.style.setProperty('min-width', '0', 'important');
      results.style.setProperty('flex', '1 1 0', 'important');
      results.style.setProperty('width', 'auto', 'important');
      results.style.setProperty('min-width', '0', 'important');
      handle.setAttribute('aria-valuemax', String(Math.round(boundedWidth(Infinity))));
      handle.setAttribute('aria-valuenow', String(Math.round(width)));
    }
    position();
  }

  let dragging = false;
  let startedClosed = false;
  function onMove(event) {
    if (!dragging) return;
    event.preventDefault();
    if (mobile.matches) {
      height = Math.max(0, event.clientY - song.getBoundingClientRect().top);
    } else {
      width = boundedWidth(event.clientX - song.getBoundingClientRect().left);
    }
    render();
  }
  function onUp() {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('is-dragging');
    if (mobile.matches) {
      if (height < 40) height = 0;
      sessionStorage.setItem(heightKey, String(height));
    } else {
      if (startedClosed && width > 0) width = Math.max(width, defaultWidth());
      else if (width <= snapWidth) width = 0;
      sessionStorage.setItem(widthKey, String(width));
    }
    render();
  }
  function onDown(event) {
    event.preventDefault();
    startedClosed = !mobile.matches && width === 0;
    dragging = true;
    handle.classList.add('is-dragging');
  }
  function onKey(event) {
    const step = event.shiftKey ? 100 : 25;
    if (event.key === 'Home') {
      mobile.matches ? height = 0 : width = 0;
    } else if (event.key === 'End') {
      mobile.matches ? height = song.scrollHeight : width = boundedWidth(Infinity);
    } else if (mobile.matches && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
      height = Math.max(0, (height ?? song.scrollHeight) + (event.key === 'ArrowDown' ? step : -step));
    } else if (!mobile.matches && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      width = event.key === 'ArrowRight' && width === 0
        ? boundedWidth(Math.max(defaultWidth(), snapWidth + step))
        : boundedWidth(width + (event.key === 'ArrowRight' ? step : -step));
      if (width <= snapWidth) width = 0;
    } else return;
    event.preventDefault();
    if (mobile.matches) sessionStorage.setItem(heightKey, String(height));
    else sessionStorage.setItem(widthKey, String(width));
    render();
  }
  handle.addEventListener('pointerdown', onDown);
  handle.addEventListener('keydown', onKey);
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
  window.addEventListener('resize', render);
  window.addEventListener('scroll', position, true);
  mobile.addEventListener('change', render);
  render();
  window.__stemPaneResizer = {
    destroy() {
      handle.remove();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      window.removeEventListener('resize', render);
      window.removeEventListener('scroll', position, true);
      mobile.removeEventListener('change', render);
    }
  };
})();
</script>
"""


def apply_style() -> None:
    st.html(STYLE)
    st.html('<script>window.__stemPaneResizer?.destroy();</script>', unsafe_allow_javascript=True)


def install_stem_pane_resizer(*, open_pane: bool = False) -> None:
    """Let the user resize or collapse the stem pane from its outer edge."""
    script = STEM_PANE_RESIZER.replace("__OPEN_PANE__", "true" if open_pane else "false")
    st.html(script, unsafe_allow_javascript=True)


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
