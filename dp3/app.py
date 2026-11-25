"""Streamlit interface for uploading, separating, and searching music stems."""

from hashlib import sha256
from html import escape
from pathlib import Path

import streamlit as st

from dp3.search import SimilarityIndex, SearchResult, query_descriptors
from dp3.stem_service import DATA_ROOT, load_or_separate, stem_cache_key
from dp3.ui import EMPTY_MAP, apply_style, waveform_player_html
from dp3.upload import (
    CLIP_SECONDS,
    CLIP_TOLERANCE_SECONDS,
    AudioUploadError,
    UploadedAudio,
    crop_audio,
    inspect_audio,
    whole_clip_audio,
)


DISPLAY_STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")


@st.cache_resource
def get_separator():
    """Load Demucs only when an uncached clip needs separation."""
    from dp3.separation import StemSeparator

    return StemSeparator()


@st.cache_resource
def _load_similarity_index(index_version: tuple[int, int]) -> SimilarityIndex:
    return SimilarityIndex.load()


def get_similarity_index() -> SimilarityIndex:
    """Reload the cached search index when the table build replaces its files."""
    folder = DATA_ROOT / "index"
    version = tuple((folder / name).stat().st_mtime_ns if (folder / name).exists() else 0
                    for name in ("embeddings.npz", "embeddings_index.csv"))
    return _load_similarity_index(version)


def add_uploads(uploads: dict[str, UploadedAudio], files: list) -> list[str]:
    """Copy newly uploaded files into the current session's memory."""
    added = []
    for file in files:
        data = file.getvalue()
        audio_id = sha256(data).hexdigest()
        if audio_id in uploads:
            continue
        try:
            upload = inspect_audio(file.name, data)
            if abs(upload.duration_seconds - CLIP_SECONDS) <= CLIP_TOLERANCE_SECONDS:
                upload.clip_bytes = whole_clip_audio(upload)
                upload.clip_start_seconds = 0.0
        except AudioUploadError as exc:
            st.error(f"{file.name}: {exc}")
            continue
        uploads[audio_id] = upload
        added.append(audio_id)
    return added


def _select_audio(audio_id: str) -> None:
    st.session_state["selected_audio_id"] = audio_id


def _select_stem(clip_key: str, stem: str) -> None:
    st.session_state[f"selected_stem_{clip_key}"] = stem


def _library_audio(upload: UploadedAudio, audio_id: str) -> tuple[bytes, str]:
    """Use a saved clip or a cached 30-second preview for library playback."""
    if upload.clip_bytes is not None:
        return upload.clip_bytes, "audio/wav"
    if upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS:
        previews: dict[str, bytes] = st.session_state.setdefault("library_previews", {})
        if audio_id not in previews:
            previews[audio_id] = crop_audio(upload, 0.0)
        return previews[audio_id], "audio/wav"
    return upload.source_bytes, upload.mime_type


def show_library(uploads: dict[str, UploadedAudio], selected_id: str | None) -> None:
    """Render the upload controls and the session's song list."""
    with st.container(key="library_panel"):
        st.html('<div class="eyebrow">LIBRARY</div>')

        with st.container(key="library_filter"):
            filter_text = st.text_input(
                "Filter songs",
                placeholder="Filter songs",
                label_visibility="collapsed",
                key="library_filter_text",
            ).casefold()

        visible = [
            (audio_id, upload)
            for audio_id, upload in uploads.items()
            if filter_text in upload.name.casefold()
        ]
        if not uploads:
            st.caption("Your uploaded songs will appear here.")
        elif not visible:
            st.caption("No songs match this filter.")

        for audio_id, upload in visible:
            row_key = "selected_song_row" if audio_id == selected_id else f"song_row_{audio_id}"
            with st.container(key=row_key, gap="xxsmall"):
                choose, duration = st.columns([3, 1], gap="small", vertical_alignment="center")
                with choose:
                    st.button(
                        Path(upload.name).stem,
                        key=f"choose_audio_{audio_id}",
                        on_click=_select_audio,
                        args=(audio_id,),
                        use_container_width=True,
                        type="tertiary",
                    )
                with duration:
                    minutes, seconds = divmod(round(upload.duration_seconds), 60)
                    st.caption(f"{minutes}:{seconds:02d}")
                try:
                    audio, audio_format = _library_audio(upload, audio_id)
                except AudioUploadError as exc:
                    st.caption(f"Preview unavailable: {exc}")
                else:
                    st.html(
                        waveform_player_html(
                            audio,
                            key=f"library_{audio_id}",
                            label=f"{Path(upload.name).stem} in library",
                            mime_type=audio_format,
                            bars=28,
                            compact=True,
                        ),
                        unsafe_allow_javascript=True,
                    )

        spacer_rem = 23.5 + 7 * len(visible)
        st.html(
            f'<div class="library-spacer" style="height:max(1rem, calc(100vh - {spacer_rem:.2f}rem))"></div>'
        )
        with st.container(key="library_drop"):
            drop_files = st.file_uploader(
                "Upload MP3 or WAV audio",
                type=["wav", "mp3"],
                accept_multiple_files=True,
                key="upload_drop_zone",
                label_visibility="collapsed",
                help="Choose files or drag and drop them here. Longer audio can be cropped to 30 seconds.",
            )
            st.html('<div class="upload-hint">or drop a clip here<br>MP3 or WAV, up to 30 s</div>')

        added = add_uploads(uploads, drop_files)
        if added:
            st.session_state["selected_audio_id"] = added[-1]
            st.rerun()


def show_song_audio(upload: UploadedAudio, audio_id: str) -> None:
    """Render playback and any required 30-second crop controls."""
    title = escape(Path(upload.name).stem)
    st.html(f'<div class="eyebrow">SONG</div><h1 class="panel-title">{title}</h1>')

    crop_preview = None
    if upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS:
        max_start = (upload.frames - CLIP_SECONDS * upload.sample_rate) / upload.sample_rate
        start = st.slider(
            "Crop start (seconds)",
            min_value=0.0,
            max_value=max_start,
            value=0.0,
            step=0.1,
            key=f"crop_start_{audio_id}",
        )
        try:
            crop_preview = crop_audio(upload, start)
        except AudioUploadError as exc:
            st.error(str(exc))
            return

    with st.container(key="mix_player"):
        audio = upload.clip_bytes or crop_preview or upload.source_bytes
        audio_format = "audio/wav" if upload.clip_bytes or crop_preview else upload.mime_type
        st.html(
            waveform_player_html(
                audio,
                key=f"mix_{audio_id}",
                label=Path(upload.name).stem,
                mime_type=audio_format,
                bars=42,
            ),
            unsafe_allow_javascript=True,
        )

    if upload.duration_seconds < CLIP_SECONDS - CLIP_TOLERANCE_SECONDS:
        st.warning("This song is shorter than 30 seconds. Upload a longer file.")
        return

    if crop_preview is not None:
        st.caption(f"30-second crop · {start:.1f}–{start + CLIP_SECONDS:.1f} s")
        if upload.clip_bytes is not None:
            st.html(
                waveform_player_html(
                    crop_preview,
                    key=f"crop_preview_{audio_id}_{start}",
                    label="Crop preview",
                    bars=42,
                ),
                unsafe_allow_javascript=True,
            )
        if st.button("Save this 30-second crop", key=f"save_crop_{audio_id}"):
            upload.clip_bytes = crop_preview
            upload.clip_start_seconds = start
            st.rerun()


def show_stems(stems: dict[str, bytes], clip_key: str) -> str | None:
    """Show six selectable stems with on-demand playback."""
    selected = st.session_state.get(f"selected_stem_{clip_key}")
    st.html('<div class="eyebrow stems-heading">STEMS</div>')

    for stem in DISPLAY_STEMS:
        row_key = "selected_stem_row" if stem == selected else f"stem_row_{stem}"
        with st.container(key=row_key, gap="xxsmall"):
            label, player = st.columns(
                [1.0, 2.6], gap="xxsmall", vertical_alignment="center"
            )
            with label:
                st.button(
                    stem.capitalize(),
                    key=f"select_stem_{clip_key}_{stem}",
                    on_click=_select_stem,
                    args=(clip_key, stem),
                    type="tertiary",
                    use_container_width=True,
                )
            with player:
                st.html(
                    waveform_player_html(
                        stems[stem],
                        key=f"stem_{clip_key}_{stem}",
                        label=f"{stem.capitalize()} stem",
                        bars=22,
                        compact=True,
                    ),
                    unsafe_allow_javascript=True,
                )
    with st.container(key="play_all_action"):
        st.button("▶  Play all", disabled=True, help="Mix playback is coming later.", use_container_width=True)
    return selected


def show_song_panel(upload: UploadedAudio, audio_id: str) -> tuple[str | None, list[SearchResult] | None]:
    with st.container(key="song_panel", gap="xxsmall"):
        show_song_audio(upload, audio_id)
        if upload.clip_bytes is None:
            return None, None

        clip_key = stem_cache_key(upload)
        stem_results: dict[str, dict[str, bytes]] = st.session_state.setdefault("stem_results", {})
        stems = stem_results.get(clip_key)
        if stems is None:
            if st.button("Separate into six stems", key=f"separate_{audio_id}", use_container_width=True):
                with st.spinner("Checking cached stems and running Demucs if needed..."):
                    try:
                        stems, _source = load_or_separate(upload, stem_results, get_separator)
                    except Exception as exc:
                        st.error(f"Could not separate this clip: {exc}")
            if stems is None:
                return None, None
        else:
            st.html('<div class="separated-status">Separated ✓</div>')

        selected_stem = show_stems(stems, clip_key)
        if selected_stem is None:
            return None, None
        searches: dict[tuple[str, str], list[SearchResult]] = st.session_state.setdefault(
            "search_results", {}
        )
        search_key = (clip_key, selected_stem)
        with st.container(key="search_action"):
            label, action = st.columns([0.48, 0.52], gap="small", vertical_alignment="center")
            with label:
                st.html(
                    '<div class="selected-label">Selected stem</div>'
                    f'<div class="selected-name">{selected_stem.capitalize()}</div>'
                )
            with action:
                if st.button("⌕  Find similar", key=f"search_button_{clip_key}", use_container_width=True):
                    with st.spinner("Extracting stem features and searching the index..."):
                        try:
                            query = query_descriptors(upload.clip_bytes, stems[selected_stem], selected_stem)
                            searches[search_key] = get_similarity_index().search(
                                selected_stem, query, upload=upload, limit=3
                            )
                        except Exception as exc:
                            st.error(f"Could not search this stem: {exc}")
        return selected_stem, searches.get(search_key)


def show_results_panel(
    upload: UploadedAudio | None,
    selected_stem: str | None = None,
    results: list[SearchResult] | None = None,
) -> None:
    with st.container(key="results_panel"):
        st.html('<div class="map-note">Visual map coming soon</div>')
        title = escape(Path(upload.name).stem) if upload is not None else ""
        heading = (
            f"Similar to: {title} - {selected_stem.capitalize()}"
            if selected_stem else "Similar sounds"
        )
        st.html('<div class="eyebrow">RESULTS</div>' f'<h2 class="panel-title">{heading}</h2>')
        if results is None:
            st.html(EMPTY_MAP)
            return
        if not results:
            st.warning("No playable matches were found in the saved index.")
            return
        for number, result in enumerate(results, 1):
            with st.container(key=f"result_card_{number}"):
                st.markdown(f"### {number}. {result.title}")
                st.caption(f"{result.dataset.upper()} · cosine similarity {result.similarity:.3f}")
                st.write("Full song")
                mix_format = "audio/mpeg" if result.mix_path.suffix.lower() == ".mp3" else "audio/wav"
                st.html(
                    waveform_player_html(
                        result.mix_path.read_bytes(),
                        key=f"result_{number}_{result.dataset}_{result.track_id}_mix",
                        label=f"{result.title} full song",
                        mime_type=mix_format,
                        bars=48,
                    ),
                    unsafe_allow_javascript=True,
                )
                st.write(f"{result.stem.capitalize()} stem")
                st.html(
                    waveform_player_html(
                        result.stem_path.read_bytes(),
                        key=f"result_{number}_{result.dataset}_{result.track_id}_{result.stem}",
                        label=f"{result.title} {result.stem} stem",
                        mime_type="audio/flac",
                        bars=48,
                    ),
                    unsafe_allow_javascript=True,
                )


def main() -> None:
    st.set_page_config(page_title="Stem Similarity Search", layout="wide")
    apply_style()

    uploads: dict[str, UploadedAudio] = st.session_state.setdefault("audio_uploads", {})
    selected_id = st.session_state.get("selected_audio_id")
    if selected_id not in uploads:
        selected_id = None

    if selected_id is None:
        library, results = st.columns([1.05, 4.3], gap=None)
        with library:
            show_library(uploads, selected_id)
        with results:
            show_results_panel(None)
        return

    library, song, results = st.columns([1.05, 1.3, 3.0], gap=None)
    with library:
        show_library(uploads, selected_id)
    with song:
        selected_stem, found = show_song_panel(uploads[selected_id], selected_id)
    with results:
        show_results_panel(uploads[selected_id], selected_stem, found)


if __name__ == "__main__":
    main()
