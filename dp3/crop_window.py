"""A fixed-length crop window over the waveform of an uploaded song."""

from io import BytesIO
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import streamlit as st

from .upload import ALAC_SUFFIXES, CLIP_SECONDS, UploadedAudio


def waveform_peaks(upload: UploadedAudio, bars: int = 96) -> list[float]:
    """Summarize the entire source waveform without retaining decoded audio."""
    peaks = np.zeros(bars, dtype=np.float32)
    if Path(upload.name).suffix.lower() in ALAC_SUFFIXES:
        with av.open(BytesIO(upload.source_bytes)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format="fltp", layout=stream.codec_context.layout, rate=upload.sample_rate
            )
            cursor = 0
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    levels = np.max(np.abs(converted.to_ndarray()), axis=0)
                    end = cursor + len(levels)
                    first = min(bars - 1, cursor * bars // upload.frames)
                    last = min(bars - 1, (end - 1) * bars // upload.frames)
                    for index in range(first, last + 1):
                        left = max(cursor, index * upload.frames // bars)
                        right = min(end, (index + 1) * upload.frames // bars)
                        if right > left:
                            peaks[index] = max(peaks[index], np.max(levels[left - cursor:right - cursor]))
                    cursor = end
    else:
        with sf.SoundFile(BytesIO(upload.source_bytes)) as audio:
            edges = np.linspace(0, audio.frames, bars + 1, dtype=int)
            for index in range(bars):
                audio.seek(int(edges[index]))
                samples = audio.read(int(edges[index + 1] - edges[index]), dtype="float32")
                if samples.size:
                    peaks[index] = np.max(np.abs(samples))
    scale = float(peaks.max())
    return (peaks / scale).tolist() if scale else peaks.tolist()


def format_crop_time(seconds: float) -> str:
    """Show a crop position as minutes and whole seconds."""
    whole_seconds = max(0, int(seconds + 0.5))
    minutes, remainder = divmod(whole_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


_CROP_WINDOW = st.components.v2.component(
    "stem_similarity_crop_window",
    html="""
        <div class="crop-window-control">
          <div class="crop-window-heading">Drag the 30-second window across the song</div>
          <div class="crop-window-track" aria-label="Full song waveform">
            <div class="crop-window-bars"></div>
            <div class="crop-window-bars crop-window-bars-active"></div>
            <div class="crop-window-selection" role="slider" tabindex="0"
                 aria-label="30-second crop position" aria-valuemin="0"></div>
          </div>
          <div class="crop-window-times"><span class="crop-window-range"></span><span class="crop-window-duration"></span></div>
        </div>
    """,
    css="""
        .crop-window-control { width: 100%; color: #f6edf4; font-family: var(--st-font); }
        .crop-window-heading { color: #cbbbc9; font-size: .82rem; margin: .2rem 0 .45rem; }
        .crop-window-track { position: relative; height: 3.4rem; cursor: pointer; touch-action: none; }
        .crop-window-bars { display: flex; align-items: center; gap: 2px; position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
        .crop-window-bars span { flex: 1 1 0; min-width: 0; border-radius: 2px; background: #6d566d; }
        .crop-window-bars-active { clip-path: inset(0 var(--right) 0 var(--left)); }
        .crop-window-bars-active span { background: #f461af; }
        .crop-window-selection { position: absolute; top: 0; bottom: 0; box-sizing: border-box;
          border: 2px solid #ff93c9; border-radius: .35rem; background: #f461af22;
          box-shadow: 0 0 0 1px #160f18; cursor: grab; touch-action: none; }
        .crop-window-selection:active { cursor: grabbing; }
        .crop-window-selection:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }
        .crop-window-times { display: flex; justify-content: space-between; gap: .4rem;
          color: #c6b3c5; font-size: .78rem; font-variant-numeric: tabular-nums; margin-top: .28rem; }
    """,
    js="""
        export default function({ parentElement, data, setStateValue }) {
          const root = parentElement.querySelector('.crop-window-control');
          const track = root.querySelector('.crop-window-track');
          const selection = root.querySelector('.crop-window-selection');
          const normal = root.querySelector('.crop-window-bars');
          const active = root.querySelector('.crop-window-bars-active');
          const range = root.querySelector('.crop-window-range');
          const total = root.querySelector('.crop-window-duration');
          const duration = Number(data.duration);
          const windowSeconds = Number(data.windowSeconds);
          const maxStart = Math.max(0, duration - windowSeconds);
          const clamp = value => Math.max(0, Math.min(maxStart, value));
          let start = clamp(Number(data.start) || 0);
          let dragOffset = 0;
          const time = value => {
            const wholeSeconds = Math.max(0, Math.round(value));
            const minutes = String(Math.floor(wholeSeconds / 60)).padStart(2, '0');
            const seconds = String(wholeSeconds % 60).padStart(2, '0');
            return `${minutes}:${seconds}`;
          };
          const render = () => {
            const left = start / duration * 100;
            const width = windowSeconds / duration * 100;
            selection.style.left = `${left}%`;
            selection.style.width = `${width}%`;
            active.style.setProperty('--left', `${left}%`);
            active.style.setProperty('--right', `${100 - left - width}%`);
            selection.setAttribute('aria-valuemax', String(maxStart.toFixed(1)));
            selection.setAttribute('aria-valuenow', String(start.toFixed(1)));
            selection.setAttribute('aria-valuetext', `${time(start)} to ${time(start + windowSeconds)}`);
            range.textContent = `${time(start)}–${time(start + windowSeconds)} selected`;
            total.textContent = `${time(duration)} total`;
          };
          const heights = data.peaks.map(peak => Math.max(12, 100 * peak));
          for (const bars of [normal, active]) {
            bars.replaceChildren(...heights.map(height => {
              const bar = document.createElement('span');
              bar.style.height = `${height}%`;
              return bar;
            }));
          }
          const move = event => {
            const bounds = track.getBoundingClientRect();
            const width = bounds.width * windowSeconds / duration;
            const available = Math.max(1, bounds.width - width);
            const x = Math.max(0, Math.min(available, event.clientX - bounds.left - dragOffset));
            start = clamp(x / available * maxStart);
            render();
          };
          track.onpointerdown = event => {
            const bounds = track.getBoundingClientRect();
            const width = bounds.width * windowSeconds / duration;
            const left = bounds.width * start / duration;
            const x = event.clientX - bounds.left;
            dragOffset = x >= left && x <= left + width ? x - left : width / 2;
            track.setPointerCapture(event.pointerId);
            move(event);
          };
          track.onpointermove = event => {
            if (track.hasPointerCapture(event.pointerId)) move(event);
          };
          const finish = event => {
            if (!track.hasPointerCapture(event.pointerId)) return;
            track.releasePointerCapture(event.pointerId);
            const rounded = Math.round(start * 10) / 10;
            if (rounded !== Number(data.start)) setStateValue('start', rounded);
          };
          track.onpointerup = finish;
          track.onpointercancel = finish;
          selection.onkeydown = event => {
            const step = event.shiftKey ? .1 : 1;
            if (event.key === 'ArrowRight') start += step;
            else if (event.key === 'ArrowLeft') start -= step;
            else if (event.key === 'Home') start = 0;
            else if (event.key === 'End') start = maxStart;
            else return;
            event.preventDefault();
            start = clamp(Math.round(start * 10) / 10);
            render();
            setStateValue('start', start);
          };
          const observer = new ResizeObserver(render);
          observer.observe(track);
          render();
          return () => observer.disconnect();
        }
    """,
)


def crop_window(upload: UploadedAudio, audio_id: str, peaks: list[float]) -> float:
    """Return the selected start time for a draggable 30-second crop."""
    key = f"crop_window_{audio_id}"
    initial = upload.clip_start_seconds or 0.0
    current = st.session_state.get(key, {}).get("start", initial)
    result = _CROP_WINDOW(
        data={"peaks": peaks, "duration": upload.duration_seconds,
              "windowSeconds": CLIP_SECONDS, "start": current},
        default={"start": initial},
        on_start_change=lambda: None,
        key=key,
    )
    max_start = upload.duration_seconds - CLIP_SECONDS
    return min(max(float(result.start), 0.0), max_start)
