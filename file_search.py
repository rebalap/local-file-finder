"""
Core logic for the local photo, voice-note, video and PDF search app.

The idea in one sentence: turn every photo, voice note, video and text query
into a vector (a list of 768 numbers) using the SAME model, so that
"a dog on a beach" lands close to photos and videos of dogs on beaches, and
"shopping list" lands close to a voice note saying "pick up milk and eggs".
Then search is just "find the nearest vectors".

Pieces:
  1. load_model()     - loads Google's EmbeddingGemma 2 (text, vision and audio).
  2. get_collection() - opens the local ChromaDB vector database on disk.
  3. index_folder()   - finds photos, voice notes, videos and PDFs, skips ones
                        already indexed, embeds the rest.
  4. search()         - embeds a text query and returns the closest matches.

Everything runs on this machine. The model is downloaded from Hugging Face once,
then cached; no photos, recordings, videos or queries are ever sent anywhere.
"""

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import av
import chromadb
import numpy as np
import torch
from pypdf import PdfReader
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sentence_transformers import SentenceTransformer

# Teach Pillow how to open .heic files (the default iPhone photo format).
register_heif_opener()

# pypdf prints warnings for damaged PDFs; we report those files ourselves.
logging.getLogger("pypdf").setLevel(logging.ERROR)

MODEL_NAME = "google/embeddinggemma-2"
DB_PATH = Path(__file__).parent / "file_index"  # where ChromaDB saves its files
COLLECTION_NAME = "media"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".opus"}  # .m4a = iPhone Voice Memos
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}  # .mov = iPhone videos
DOCUMENT_EXTENSIONS = {".pdf"}

BATCH_SIZE = 16  # how many images to embed at once; bigger = faster but more memory
SAMPLE_RATE = 16_000  # the model expects audio as 16 kHz mono

# Voice notes are embedded as overlapping 10-second windows, one starting every
# 2.5 seconds. In testing this found the right moment in a recording to within
# about 1.5 seconds (30-second pieces could be off by up to 30 seconds).
WINDOW_SECONDS = 10
STEP_SECONDS = 2.5

# Videos: the model card recommends 1 frame per second. The model's video
# processor takes at most 32 frames (max_frames in its config), so clips longer
# than 32 seconds get 32 frames spread evenly instead.
MAX_VIDEO_FRAMES = 32
VIDEO_FRAME_SIZE = 768  # shrink frames to at most this many pixels across (saves RAM on 4K video)

# PDFs: each page's text is split into passages of about 120 words, each
# overlapping the previous one by 40 words, so a search result can point to
# a specific paragraph (and a sentence cut at a boundary is whole in the next).
PASSAGE_WORDS = 120
PASSAGE_OVERLAP = 40

# Bump this when the way files are embedded changes. Entries made with an older
# version are treated as "not indexed", so the next index run redoes them.
INDEX_VERSION = 2

# "Show match area" for photos: the photo is cut into a 3x3 grid of overlapping
# tiles, each half the photo's width and height. A box is drawn only if the
# tiles' scores differ by at least this much; otherwise the match isn't in any
# one place (e.g. a whole-scene query, or something not in the photo at all).
# 0.045 was chosen by testing object, scene and "not in the photo" queries.
MIN_TILE_SPREAD = 0.045

# How much detail the model uses per tile. Photos are embedded at the model's
# default of 280 "tokens"; tiles use 140, which is twice as fast (about 2 s per
# photo instead of 4) and in testing picked an equally good tile overall.
TILE_TOKENS = 140

# The tile setting is changed on the shared model for a moment, so this lock
# stops photo indexing from running at the same time and using it by mistake.
_image_detail_lock = threading.Lock()


def load_model() -> SentenceTransformer:
    """Load EmbeddingGemma 2 with its text, vision and audio parts (740M parameters)."""
    if torch.backends.mps.is_available():
        # Apple Silicon GPU. bfloat16 halves memory. Note: the model card warns
        # NOT to use float16, which produces broken (NaN) embeddings.
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    settings = dict(device=device, model_kwargs={"torch_dtype": dtype})
    try:
        # Use the copy already downloaded to ~/.cache/huggingface, without
        # contacting Hugging Face at all.
        return SentenceTransformer(MODEL_NAME, local_files_only=True, **settings)
    except OSError:
        # Very first run: the model isn't on disk yet, so download it once.
        return SentenceTransformer(MODEL_NAME, **settings)


def get_collection():
    """Open (or create) the vector database stored in ./file_index.

    We tell Chroma to use cosine distance, the standard way to compare
    embeddings: it measures the angle between vectors, ignoring their length.
    """
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=chromadb.Settings(anonymized_telemetry=False),  # no usage stats sent out
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def media_type(path: str) -> str | None:
    """Return "photo", "voice", "video", "document" or None (unsupported) from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "photo"
    if suffix in AUDIO_EXTENSIONS:
        return "voice"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    return None


def find_media(folder: str) -> list[str]:
    """Return the full paths of all supported files in a folder (and its subfolders)."""
    paths = []
    for root, dirs, files in os.walk(folder):
        # Don't descend into hidden folders (e.g. .venv) or our own database folder.
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != DB_PATH.name]
        for name in files:
            if media_type(name):
                paths.append(str(Path(root, name).resolve()))
    return sorted(paths)


def file_mtime(path: str) -> int:
    """When the file was last modified, in whole milliseconds.

    We store it as an integer because ChromaDB slightly rounds decimal numbers,
    which would make an unchanged file look "modified" and get re-indexed.
    """
    return os.stat(path).st_mtime_ns // 1_000_000


def load_image(path: str) -> Image.Image:
    """Open an image as RGB, rotated the right way up (phones store rotation in EXIF)."""
    image = ImageOps.exif_transpose(Image.open(path))
    return image.convert("RGB")


def load_audio(path: str) -> np.ndarray:
    """Decode any audio file (e.g. iPhone .m4a) into 16 kHz mono samples.

    PyAV bundles its own decoders, so this works without installing ffmpeg.
    The resampler converts whatever the file uses (e.g. 48 kHz stereo) into
    the 16 kHz mono floats the model expects.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    pieces = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            pieces += [f.to_ndarray()[0] for f in resampler.resample(frame)]
    pieces += [f.to_ndarray()[0] for f in resampler.resample(None)]  # flush the last bit
    if not pieces:
        raise ValueError("no audio found")
    return np.concatenate(pieces)


def split_audio(samples: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Cut a recording into overlapping 10-second windows, one every 2.5 seconds.

    Returns a list of (start_seconds, samples). A voice note of 10 seconds or
    less stays as one piece. The windows overlap so that whatever moment you
    search for sits near the start of at least one window, which lets search
    point to where in the recording it was said. Near the end, windows are
    allowed to be shorter (down to 5 s), so the last few seconds get their own
    window instead of being mixed in with what came before.
    """
    size = WINDOW_SECONDS * SAMPLE_RATE
    step = int(STEP_SECONDS * SAMPLE_RATE)
    if len(samples) <= size:
        return [(0.0, samples)]

    last_start = len(samples) - size // 2  # keep every window at least 5 s long
    return [
        (start / SAMPLE_RATE, samples[start : start + size])
        for start in range(0, last_start, step)
    ]


def _index_photos(paths: list[str], model, collection) -> list[str]:
    """Embed one batch of photos and save them. Returns paths that couldn't be read."""
    images, good_paths, failed = [], [], []
    for path in paths:
        try:
            images.append(load_image(path))
            good_paths.append(path)
        except Exception:  # corrupt or unreadable file
            failed.append(path)

    if images:
        # Turn the images into vectors. normalize=True makes every vector
        # length 1, so cosine similarity is a fair comparison.
        with _image_detail_lock:  # make sure tiles aren't changing the detail setting right now
            embeddings = model.encode([{"image": img} for img in images], normalize_embeddings=True)
        collection.upsert(
            ids=good_paths,  # a photo's ID is just its path
            embeddings=embeddings.tolist(),
            metadatas=[
                {"path": p, "mtime": file_mtime(p), "type": "photo", "start": 0.0,
                 "version": INDEX_VERSION}
                for p in good_paths
            ],
        )
    return failed


def _index_voice_note(path: str, samples: np.ndarray, model, collection) -> None:
    """Embed one voice note (all its 10-second windows) and save it."""
    chunks = split_audio(samples)
    embeddings = model.encode(
        [{"audio": {"array": samples, "sampling_rate": SAMPLE_RATE}} for _start, samples in chunks],
        normalize_embeddings=True,
        batch_size=8,
    )
    mtime = file_mtime(path)
    collection.upsert(
        ids=[f"{path}#{start:.1f}" for start, _samples in chunks],  # e.g. ".../memo.m4a#27.5"
        embeddings=embeddings.tolist(),
        metadatas=[
            {"path": path, "mtime": mtime, "type": "voice", "start": start, "version": INDEX_VERSION}
            for start, _samples in chunks
        ],
    )


def load_video_frames(path: str) -> list[Image.Image]:
    """Take one frame per second from a video (or 50 spread evenly if it's longer).

    PyAV (the same library that reads voice notes) decodes the video. Each
    frame is turned upright (phones store portrait videos sideways plus a
    rotation flag) and shrunk, since the model resizes them anyway.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0

        # Which moments to take a frame from: every second, or evenly spread.
        if duration <= MAX_VIDEO_FRAMES:
            wanted = [float(t) for t in range(int(duration) + 1)]
        else:
            wanted = [i * duration / MAX_VIDEO_FRAMES for i in range(MAX_VIDEO_FRAMES)]

        frames = []
        for frame in container.decode(stream):
            if not wanted:
                break  # got every frame we wanted
            if frame.time is not None and frame.time >= wanted[0]:
                image = frame.to_image()
                if frame.rotation:
                    image = image.rotate(frame.rotation, expand=True)
                image.thumbnail((VIDEO_FRAME_SIZE, VIDEO_FRAME_SIZE))
                frames.append(image)
                while wanted and frame.time >= wanted[0]:
                    wanted.pop(0)

    if not frames:
        raise ValueError("no video frames found")
    return frames


def _index_video(path: str, frames: list[Image.Image], model, collection) -> None:
    """Embed one video clip, as a whole, and save it."""
    video = np.stack([np.asarray(frame) for frame in frames])  # (frames, height, width, 3)
    embedding = model.encode(
        [{"video": {
            "array": video,
            # We already picked the frames, so tell the model "1 frame per
            # second" for exactly this many frames; it then uses all of them.
            "video_metadata": {"fps": 1.0, "total_num_frames": len(frames), "duration": len(frames)},
        }}],
        normalize_embeddings=True,
    )
    collection.upsert(
        ids=[path],
        embeddings=embedding.tolist(),
        metadatas=[{"path": path, "mtime": file_mtime(path), "type": "video", "start": 0.0,
                    "version": INDEX_VERSION}],
    )


def load_pdf_passages(path: str) -> tuple[str, list[tuple[int, str]]]:
    """Read a PDF's text and split it into passages of about 120 words.

    Returns (title, [(page_number, passage_text), ...]). Passages never cross
    a page boundary, so every result can say which page it's on. Scanned PDFs
    (pictures of pages, with no real text) raise an error and are reported as
    unreadable.
    """
    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or Path(path).stem
    step = PASSAGE_WORDS - PASSAGE_OVERLAP

    passages = []
    for page_number, page in enumerate(reader.pages, start=1):
        words = (page.extract_text() or "").split()
        # Start a passage every 80 words, as long as there are words left that
        # the previous passage didn't already cover.
        for start in range(0, max(1, len(words) - PASSAGE_OVERLAP), step):
            text = " ".join(words[start : start + PASSAGE_WORDS])
            if text:
                passages.append((page_number, text))

    if not passages:
        raise ValueError("no text found (scanned PDF?)")
    return title, passages


def _index_document(path: str, title: str, passages: list[tuple[int, str]], model, collection) -> None:
    """Embed every passage of one PDF and save it."""
    # The model card's format for documents is "title: ... | text: ...".
    # Giving the PDF's real title adds context to every passage.
    embeddings = model.encode(
        [text for _page, text in passages],
        prompt=f"title: {title} | text: ",
        normalize_embeddings=True,
    )
    mtime = file_mtime(path)
    collection.upsert(
        ids=[f"{path}#{i}" for i in range(len(passages))],  # e.g. ".../notes.pdf#7"
        embeddings=embeddings.tolist(),
        metadatas=[
            {"path": path, "mtime": mtime, "type": "document", "start": 0.0,
             "page": page, "text": text, "version": INDEX_VERSION}
            for page, text in passages
        ],
    )


def _remove_missing_files(folder: str, present: set[str], collection) -> int:
    """Delete entries for files under `folder` that are no longer there. Returns how many files."""
    prefix = str(Path(folder).resolve()) + os.sep
    stored = {m["path"] for m in collection.get(include=["metadatas"])["metadatas"]}
    missing = [p for p in stored if p.startswith(prefix) and p not in present]
    if missing:
        collection.delete(where={"path": {"$in": missing}})
    return len(missing)


def index_folder(folder: str, model, collection, progress_callback=None) -> dict:
    """Embed every new or changed photo, voice note, video and PDF in `folder` and save it.

    How "skip already indexed" works: every entry stores its file's path,
    last-modified time and INDEX_VERSION. If a path is already there with the
    same modified time and the current version, we skip it. If the file was
    edited since (or was indexed the old way), we delete its old entries and
    embed it again. Files that were in this folder last time but have since
    been deleted or moved are removed from the database, so they stop showing
    up in results.

    progress_callback(done, total) is called as files are processed so the UI
    can show a progress bar.
    """
    all_paths = find_media(folder)
    removed = _remove_missing_files(folder, set(all_paths), collection)

    # Look up which of these files are already in the database, when they were
    # last modified, and with which version of this code.
    # (A voice note can have several entries, one per window.)
    existing = {}
    if all_paths:
        found = collection.get(where={"path": {"$in": all_paths}}, include=["metadatas"])
        existing = {m["path"]: (m["mtime"], m.get("version")) for m in found["metadatas"]}

    to_index = [p for p in all_paths if existing.get(p) != (file_mtime(p), INDEX_VERSION)]
    skipped = len(all_paths) - len(to_index)

    # Files that changed since last time: remove their old entries first.
    changed = [p for p in to_index if p in existing]
    if changed:
        collection.delete(where={"path": {"$in": changed}})

    photos = [p for p in to_index if media_type(p) == "photo"]
    voice_notes = [p for p in to_index if media_type(p) == "voice"]
    videos = [p for p in to_index if media_type(p) == "video"]
    documents = [p for p in to_index if media_type(p) == "document"]
    failed, done = [], 0

    def report():
        if progress_callback:
            progress_callback(done, len(to_index))

    # Photos are embedded in batches; everything else one file at a time.
    for start in range(0, len(photos), BATCH_SIZE):
        batch = photos[start : start + BATCH_SIZE]
        failed += _index_photos(batch, model, collection)
        done += len(batch)
        report()

    for path in voice_notes:
        try:
            samples = load_audio(path)
        except Exception:  # corrupt, empty or unreadable recording
            failed.append(path)
        else:
            _index_voice_note(path, samples, model, collection)
        done += 1
        report()

    for path in videos:
        try:
            frames = load_video_frames(path)
        except Exception:  # corrupt, empty or unreadable video
            failed.append(path)
        else:
            _index_video(path, frames, model, collection)
        done += 1
        report()

    for path in documents:
        try:
            title, passages = load_pdf_passages(path)
        except Exception:  # corrupt, encrypted or scanned (no text) PDF
            failed.append(path)
        else:
            _index_document(path, title, passages, model, collection)
        done += 1
        report()

    return {
        "found": len(all_paths),
        "photos": sum(media_type(p) == "photo" for p in all_paths),
        "voice_notes": sum(media_type(p) == "voice" for p in all_paths),
        "videos": sum(media_type(p) == "video" for p in all_paths),
        "documents": sum(media_type(p) == "document" for p in all_paths),
        "indexed": len(to_index) - len(failed),
        "skipped": skipped,
        "removed": removed,
        "failed": failed,
    }


def search(query: str, model, collection, top_k: int = 12, only: str | None = None) -> list[dict]:
    """Return the `top_k` files whose embeddings are closest to the text query.

    only: "photo", "voice", "video" or "document" to search one kind of file,
    or None for all. Each result is {"path", "score", "type", "start", "page",
    "text"}: "start" is the second the best-matching window of a voice note
    begins (0 otherwise); "page" and "text" are the best-matching passage of a
    PDF (None otherwise).
    """
    where = {"type": only} if only else None
    total = collection.count()
    if total == 0:
        return []

    # "SearchQuery" is the task prompt the model card recommends for search
    # queries; it tells the model this text is a question looking for matches.
    query_embedding = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)

    # One voice note or PDF can have many entries (windows, passages), and we
    # only want its best one. So we ask for extra matches, and if one long file
    # still filled most of them, ask again for more until we have `top_k`
    # different files (or have looked at everything).
    n_results = top_k * 5
    while True:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, total),
            where=where,
            include=["metadatas", "distances"],
        )
        best_per_file = {}
        for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
            if meta["path"] not in best_per_file:  # results arrive best-first
                best_per_file[meta["path"]] = {
                    "path": meta["path"],
                    # Chroma returns cosine *distance* (0 = identical). Similarity = 1 - distance.
                    "score": 1 - distance,
                    "type": meta["type"],
                    "start": meta["start"],
                    "page": meta.get("page"),
                    "text": meta.get("text"),
                }
        if len(best_per_file) >= top_k or n_results >= total:
            return list(best_per_file.values())[:top_k]
        n_results *= 4


def count_files(collection) -> dict:
    """How many files of each kind are indexed (counting files, not pieces)."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    paths = {(meta["type"], meta["path"]) for meta in metadatas}
    return {
        "photo": sum(kind == "photo" for kind, _path in paths),
        "voice": sum(kind == "voice" for kind, _path in paths),
        "video": sum(kind == "video" for kind, _path in paths),
        "document": sum(kind == "document" for kind, _path in paths),
    }


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """A 3x3 grid of overlapping tiles, each half the photo's width and height.

    Tiles start at 0%, 25% and 50% across (and down), so neighbouring tiles
    overlap by half. Each box is (left, top, right, bottom) in pixels.
    """
    return [
        (int(x * width / 4), int(y * height / 4),
         int(x * width / 4 + width / 2), int(y * height / 4 + height / 2))
        for y in range(3)
        for x in range(3)
    ]


@contextmanager
def image_detail(model, tokens: int):
    """Temporarily embed images with a different level of detail (tokens per image).

    sentence-transformers has no per-call option for this, so we change the
    model's image processor setting and always put it back afterwards.
    """
    processor = model[0].processor.image_processor
    with _image_detail_lock:
        original = processor.max_soft_tokens
        processor.max_soft_tokens = tokens
        try:
            yield
        finally:
            processor.max_soft_tokens = original


def find_match_area(path: str, query: str, model) -> dict:
    """Find which part of a photo best matches the query (for "Show match area").

    The model gives one vector per image, so it can't say *where* a match is.
    Instead we embed 9 overlapping tiles of the photo separately and compare
    each one with the query. The best tile is the area to highlight, but only
    if the tiles' scores really differ (see MIN_TILE_SPREAD).

    Takes about 2 seconds per photo, so it only runs on the photos you ask
    for (or the top 4, with auto-highlight on).
    """
    image = load_image(path)
    boxes = tile_boxes(*image.size)
    with image_detail(model, TILE_TOKENS):
        tile_vectors = model.encode(
            [{"image": image.crop(box)} for box in boxes], normalize_embeddings=True
        )
    query_vector = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
    scores = tile_vectors @ query_vector  # cosine similarity of each tile with the query

    best = int(np.argmax(scores))
    spread = float(scores.max() - scores.min())
    return {
        "box": boxes[best],
        "score": float(scores[best]),
        "spread": spread,
        "stands_out": spread >= MIN_TILE_SPREAD,
    }
