"""Streamlit entry point for Stem Similarity Search."""

from hashlib import sha256

import streamlit as st

from dp3.search import SimilarityIndex, SearchResult, query_descriptors
from dp3.stem_service import STEM_NAMES, load_or_separate, stem_cache_key
from dp3.upload import (
    CLIP_SECONDS,
    CLIP_TOLERANCE_SECONDS,
    AudioUploadError,
    UploadedAudio,
    crop_audio,
    inspect_audio,
    whole_clip_audio,
)


@st.cache_resource
def get_separator():
    """Load Demucs once and share its read-only model between app reruns."""
    from dp3.separation import StemSeparator

    return StemSeparator()


@st.cache_resource
def get_similarity_index() -> SimilarityIndex:
    """Read the offline embeddings once for all app reruns."""
    return SimilarityIndex.load()


def add_uploads(uploads: dict[str, UploadedAudio]) -> list[str]:
    """Copy new files from the uploader into this session's in-memory collection."""
    added = []
    selected_files = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="Choose files or drag and drop them here. Each clip must be 30 seconds.",
    )
    for file in selected_files:
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


def show_selected_audio(upload: UploadedAudio, audio_id: str) -> None:
    """Show source playback and require an explicit crop for longer files."""
    st.subheader(upload.name)
    st.caption(f"Original audio · {upload.duration_seconds:.2f} seconds")
    st.audio(upload.source_bytes, format=upload.mime_type)

    if upload.duration_seconds < CLIP_SECONDS - CLIP_TOLERANCE_SECONDS:
        st.warning("This file is shorter than 30 seconds. Upload a longer file to make a clip.")
        return

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
            preview_bytes = crop_audio(upload, start)
        except AudioUploadError as exc:
            st.error(str(exc))
        else:
            st.caption(f"Preview · {start:.1f}–{start + CLIP_SECONDS:.1f} seconds")
            st.audio(preview_bytes, format="audio/wav")
            if st.button("Save this 30-second crop", key=f"save_crop_{audio_id}"):
                upload.clip_bytes = preview_bytes
                upload.clip_start_seconds = start

    if upload.clip_bytes is not None:
        st.success("30-second clip saved in this session's memory.")
        st.caption(f"Saved clip · starts at {upload.clip_start_seconds:.1f} seconds")
        st.audio(upload.clip_bytes, format="audio/wav")


def show_stems(stems: dict[str, bytes]) -> None:
    st.subheader("Separated stems")
    for name in STEM_NAMES:
        st.write(name.capitalize())
        st.audio(stems[name], format="audio/wav")


def show_search_results(results: list[SearchResult]) -> None:
    if not results:
        st.warning("No playable matches were found in the saved index.")
        return
    st.subheader("Most similar songs")
    for number, result in enumerate(results, 1):
        st.markdown(f"### {number}. {result.title}")
        st.caption(f"{result.dataset.upper()} · cosine similarity {result.similarity:.3f}")
        st.write("Full song")
        mix_format = "audio/mpeg" if result.mix_path.suffix.lower() == ".mp3" else "audio/wav"
        st.audio(str(result.mix_path), format=mix_format)
        st.write(f"{result.stem.capitalize()} stem")
        st.audio(str(result.stem_path), format="audio/flac")


def show_search(upload: UploadedAudio, clip_key: str, stems: dict[str, bytes]) -> None:
    st.subheader("Find similar stems")
    selected_stem = st.selectbox("Stem to search", STEM_NAMES, key=f"search_stem_{clip_key}")
    searches: dict[tuple[str, str], list[SearchResult]] = st.session_state.setdefault("search_results", {})
    search_key = (clip_key, selected_stem)
    if st.button("Find top 3 similar songs", key=f"search_button_{clip_key}"):
        with st.spinner("Extracting stem features and searching the index..."):
            try:
                query = query_descriptors(upload.clip_bytes, stems[selected_stem], selected_stem)
                searches[search_key] = get_similarity_index().search(
                    selected_stem, query, upload=upload, limit=3
                )
            except Exception as exc:
                st.error(f"Could not search this stem: {exc}")
    if search_key in searches:
        show_search_results(searches[search_key])


def main() -> None:
    st.set_page_config(page_title="Stem Similarity Search")
    st.title("Stem Similarity Search")
    st.write("Upload a WAV or MP3 file. Files longer than 30 seconds need a crop.")

    uploads: dict[str, UploadedAudio] = st.session_state.setdefault("audio_uploads", {})
    added = add_uploads(uploads)
    if not uploads:
        return

    if added:
        st.session_state["selected_audio_id"] = added[-1]
    selected_id = st.selectbox(
        "Uploaded audio",
        options=list(uploads),
        format_func=lambda audio_id: f"{uploads[audio_id].name} ({uploads[audio_id].duration_seconds:.1f} s)",
        key="selected_audio_id",
    )
    selected_upload = uploads[selected_id]
    show_selected_audio(selected_upload, selected_id)

    if selected_upload.clip_bytes is None:
        return

    stem_results: dict[str, dict[str, bytes]] = st.session_state.setdefault("stem_results", {})
    key = stem_cache_key(selected_upload)
    stems = stem_results.get(key)
    if stems is None and st.button("Separate into six stems", key=f"separate_{selected_id}"):
        with st.spinner("Checking cached stems and running Demucs if needed..."):
            try:
                stems, source = load_or_separate(selected_upload, stem_results, get_separator)
            except Exception as exc:
                st.error(f"Could not separate this clip: {exc}")
            else:
                st.success(f"Stems loaded from {source} and saved in session memory.")
    if stems is not None:
        show_stems(stems)
        show_search(selected_upload, key, stems)


if __name__ == "__main__":
    main()
