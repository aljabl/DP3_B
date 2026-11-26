"""Streamlit interface for uploading, separating, and searching music stems."""

from hashlib import sha256
from html import escape
from pathlib import Path

import streamlit as st

from dp3.crop_window import crop_window, format_crop_time, waveform_peaks
from dp3.save_audio import save_and_index_upload
from dp3.search import SimilarityIndex, SearchResult, query_descriptors
from dp3.stem_service import DATA_ROOT, load_or_separate, stem_cache_key
from dp3.ui import EMPTY_MAP, apply_style, install_stem_pane_resizer, waveform_player_html
from dp3.upload import (
    CLIP_SECONDS,
    CLIP_TOLERANCE_SECONDS,
    AudioUploadError,
    UploadedAudio,
    crop_audio,
    inspect_audio,
    source_audio_for_playback,
    whole_clip_audio,
)


DISPLAY_STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")


@st.cache_resource
def get_separator():
    """Load Demucs only when an uncached clip needs separation."""
    from dp3.separation import StemSeparator

    return StemSeparator()


@st.cache_resource
def _load_similarity_index(index_version: tuple[int, ...]) -> SimilarityIndex:
    return SimilarityIndex.load()


def get_similarity_index() -> SimilarityIndex:
    """Reload when the table or saved uploads change."""
    folder = DATA_ROOT / "index"
    version = tuple((folder / name).stat().st_mtime_ns if (folder / name).exists() else 0
                    for name in ("embeddings.npz", "embeddings_index.csv"))
    user_index = folder / "user_added"
    return _load_similarity_index((*version, user_index.stat().st_mtime_ns if user_index.exists() else 0))


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
    st.session_state["open_stem_pane"] = True


def _select_stem(clip_key: str, stem: str) -> None:
    st.session_state[f"selected_stem_{clip_key}"] = stem


def display_title(upload: UploadedAudio, audio_id: str) -> str:
    """Use the edited save title throughout the current session."""
    edited = st.session_state.get(f"save_title_{audio_id}", "").strip()
    return edited or Path(upload.name).stem


def _library_audio(upload: UploadedAudio, audio_id: str) -> tuple[bytes, str]:
    """Use a saved clip or a cached 30-second preview for library playback."""
    if upload.clip_bytes is not None:
        return upload.clip_bytes, "audio/wav"
    if upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS:
        previews: dict[str, bytes] = st.session_state.setdefault("library_previews", {})
        if audio_id not in previews:
            previews[audio_id] = crop_audio(upload, 0.0)
        return previews[audio_id], "audio/wav"
    return source_audio_for_playback(upload)


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
            if filter_text in display_title(upload, audio_id).casefold()
        ]
        with st.container(key="library_song_list"):
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
                            display_title(upload, audio_id),
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
                                label=f"{display_title(upload, audio_id)} in library",
                                mime_type=audio_format,
                                bars=28,
                                compact=True,
                            ),
                            unsafe_allow_javascript=True,
                        )

        with st.container(key="library_drop"):
            drop_files = st.file_uploader(
                "Upload WAV, MP3, or ALAC audio",
                type=["wav", "mp3", "m4a", "mp4", "caf"],
                accept_multiple_files=True,
                key="upload_drop_zone",
                label_visibility="collapsed",
                help="Choose or drop WAV, MP3, or ALAC files. Longer audio can be cropped to 30 seconds.",
            )
            st.html('<div class="upload-hint">or drop a clip here<br>WAV, MP3, or ALAC</div>')

        added = add_uploads(uploads, drop_files)
        if added:
            st.session_state["selected_audio_id"] = added[-1]
            st.session_state["open_stem_pane"] = True
            st.rerun()


def show_song_audio(upload: UploadedAudio, audio_id: str) -> bool:
    """Render playback and return whether the selected crop is saved."""
    title = escape(display_title(upload, audio_id))
    st.html(f'<div class="eyebrow">SONG</div><h1 class="panel-title">{title}</h1>')

    crop_preview = None
    selected_crop_saved = upload.clip_bytes is not None
    if upload.duration_seconds > CLIP_SECONDS + CLIP_TOLERANCE_SECONDS:
        try:
            waveforms = st.session_state.setdefault("crop_waveforms", {})
            if audio_id not in waveforms:
                waveforms[audio_id] = waveform_peaks(upload)
            start = crop_window(upload, audio_id, waveforms[audio_id])
            crop_preview = crop_audio(upload, start)
        except AudioUploadError as exc:
            st.error(str(exc))
            return False
        selected_crop_saved = (
            upload.clip_bytes is not None
            and upload.clip_start_seconds is not None
            and abs(start - upload.clip_start_seconds) < 0.05
        )

    with st.container(key="mix_player"):
        try:
            if crop_preview or upload.clip_bytes:
                audio, audio_format = crop_preview or upload.clip_bytes, "audio/wav"
            else:
                audio, audio_format = source_audio_for_playback(upload)
        except AudioUploadError as exc:
            st.error(f"Could not play this song: {exc}")
            return False
        st.html(
            waveform_player_html(
                audio,
                key=f"mix_{audio_id}",
                label=display_title(upload, audio_id),
                mime_type=audio_format,
                bars=42,
            ),
            unsafe_allow_javascript=True,
        )

    if upload.duration_seconds < CLIP_SECONDS - CLIP_TOLERANCE_SECONDS:
        st.warning("This song is shorter than 30 seconds. Upload a longer file.")
        return False

    if crop_preview is not None:
        st.caption(
            f"30-second crop · {format_crop_time(start)}–{format_crop_time(start + CLIP_SECONDS)}"
        )
        if st.button("Save this 30-second crop", key=f"save_crop_{audio_id}",
                     disabled=selected_crop_saved):
            upload.clip_bytes = crop_preview
            upload.clip_start_seconds = start
            st.rerun()

    with st.container(key="save_song_action"):
        title_to_save = st.text_input(
            "Title for saved audio",
            value=Path(upload.name).stem,
            key=f"save_title_{audio_id}",
            help="The title becomes the filename in data/user_added.",
            persist_state="session",
        )
        if st.button(
            "Save song to disk",
            key=f"save_song_{audio_id}",
            disabled=not selected_crop_saved,
            help="Save the selected 30-second crop first." if not selected_crop_saved else None,
            width="stretch",
        ):
            with st.spinner("Saving the song, separating six stems, and indexing them..."):
                try:
                    stem_results = st.session_state.setdefault("stem_results", {})
                    saved_path = save_and_index_upload(
                        upload, title_to_save, stem_results, get_separator
                    )
                except FileExistsError:
                    st.warning("A song with that title already exists in data/user_added. Choose another title.")
                except Exception as exc:
                    st.error(f"Could not save and index this song: {exc}")
                else:
                    if upload.duration_seconds > CLIP_SECONDS and saved_path.suffix == ".wav":
                        upload.clip_bytes = saved_path.read_bytes()
                    st.success(f"Saved {saved_path.name}, its six stems, and its search entry.")
    return selected_crop_saved


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
        if not show_song_audio(upload, audio_id):
            return None, None

        clip_key = stem_cache_key(upload)
        stem_results: dict[str, dict[str, bytes]] = st.session_state.setdefault("stem_results", {})
        stems = stem_results.get(clip_key)
        if stems is None:
            if st.button("Separate into six stems", key=f"separate_{audio_id}", use_container_width=True):
                with st.spinner("Checking cached stems and running Demucs if needed..."):
                    try:
                        load_or_separate(upload, stem_results, get_separator)
                    except Exception as exc:
                        st.error(f"Could not separate this clip: {exc}")
                    else:
                        st.rerun()
            return None, None

        st.html('<div class="separated-status">Separated ✓</div>')

        selected_stem = show_stems(stems, clip_key)
        if selected_stem is None:
            return None, None
        searches: dict[tuple[str, str, bool], list[SearchResult]] = st.session_state.setdefault(
            "search_results", {}
        )
        with st.container(key="search_action", gap="small"):
            st.html(
                '<div class="selected-label">Selected stem</div>'
                f'<div class="selected-name">{selected_stem.capitalize()}</div>'
            )
            action, scope = st.columns([1, 1.15], gap="small", vertical_alignment="center")
            with scope:
                all_stems = st.checkbox(
                    "Search all stem types",
                    key=f"search_all_stems_{clip_key}",
                    help="Include vocals, drums, bass, guitar, piano, and other stems in the results.",
                    persist_state="session",
                    wrap=True,
                )
            search_key = (clip_key, selected_stem, all_stems)
            with action:
                if st.button("⌕  Find similar", key=f"search_button_{clip_key}", use_container_width=True):
                    with st.spinner("Extracting stem features and searching the index..."):
                        try:
                            query = query_descriptors(upload.clip_bytes, stems[selected_stem], selected_stem)
                            searches[search_key] = get_similarity_index().search(
                                selected_stem, query, upload=upload, limit=3,
                                all_stems=all_stems,
                            )
                        except Exception as exc:
                            st.error(f"Could not search this stem: {exc}")
        return selected_stem, searches.get(search_key)


def show_results_panel(
    upload: UploadedAudio | None,
    selected_stem: str | None = None,
    results: list[SearchResult] | None = None,
    *,
    audio_id: str | None = None,
) -> None:
    with st.container(key="results_panel"):
        st.html('<div class="map-note">Visual map coming soon</div>')
        title = escape(display_title(upload, audio_id)) if upload is not None and audio_id else ""
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
                st.caption(
                    f"{result.dataset.upper()} · {result.stem.capitalize()} stem · "
                    f"cosine similarity {result.similarity:.3f}"
                )
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
        show_results_panel(uploads[selected_id], selected_stem, found, audio_id=selected_id)
    install_stem_pane_resizer(open_pane=st.session_state.pop("open_stem_pane", False))


if __name__ == "__main__":
    main()
