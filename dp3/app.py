"""Streamlit entry point for Stem Similarity Search."""

from hashlib import sha256

import streamlit as st

from dp3.upload import CLIP_SECONDS, AudioUploadError, UploadedAudio, crop_audio, inspect_audio


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
            if upload.frames == CLIP_SECONDS * upload.sample_rate:
                upload.clip_bytes = crop_audio(upload, 0)
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

    if upload.frames < CLIP_SECONDS * upload.sample_rate:
        st.warning("This file is shorter than 30 seconds. Upload a longer file to make a clip.")
        return

    if upload.frames > CLIP_SECONDS * upload.sample_rate:
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
    show_selected_audio(uploads[selected_id], selected_id)


if __name__ == "__main__":
    main()
