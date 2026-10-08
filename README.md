# File Finder

Search the photos, voice notes, videos **and PDFs** on your laptop by describing
them in plain English, for example "kids playing in the snow", "the voice memo about
booking the dentist", "a woman riding a bicycle" or "tips for night trains". It runs
fully offline: no cloud APIs, and your files and searches never leave your machine.

## How it works

1. **Indexing.** The app walks a folder (and its subfolders) for photos (`.jpg`,
   `.jpeg`, `.png`, `.heic`), voice notes (`.m4a` from iPhone Voice Memos, plus
   `.mp3`, `.wav`, `.ogg`, `.opus`), videos (`.mp4`, `.mov`, `.m4v`) and PDFs. Each
   file goes through Google's
   [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) model, which
   turns it into an **embedding**: a list of 768 numbers that captures what the
   photo or video shows, or what the recording or document *says*. The embeddings are saved, along with
   each file's path, in a local [ChromaDB](https://www.trychroma.com/) vector
   database (`./file_index/`).
2. **Searching.** Your text query goes through the *same* model. Because
   EmbeddingGemma 2 puts text, images, video and audio in one shared space,
   "shopping list" lands near a voice note saying "pick up milk and eggs", without
   any speech-to-text step. ChromaDB returns the 12 nearest files, ranked by cosine
   similarity (higher is a better match). You can search everything at once, or only
   photos, voice notes, videos or documents.
3. **Videos are embedded as whole clips.** One frame per second is taken from the
   clip (as the model card recommends) and the frames go into the model together,
   so it sees the clip as a video rather than separate photos. Clips longer than 32
   seconds get 32 frames spread evenly, because the model's video processor takes at
   most 32 frames (`max_frames` in its config). Portrait phone videos are turned
   upright first.
4. **PDFs are split into passages.** The text of each page is split into passages
   of about 120 words, each overlapping the previous one by 40 words, and each
   passage is embedded with the model card's document format
   (`title: <PDF title> | text: <passage>`). Search keeps the best passage per PDF
   and shows its page number and text, plus an **Open PDF** button.
5. **Voice notes play from the matching moment.** Recordings longer than 10
   seconds are embedded as overlapping 10-second windows, one starting every 2.5
   seconds, each remembering its start time. Search keeps the best window per
   file, and the player starts 2 seconds before it. In testing this landed within
   about 1.5 seconds of where the topic began.
6. **"Show match area" highlights where in a photo the match is.** The model gives
   one vector per photo, so it can't say *where* something is. When you click the
   button, the photo is cut into 9 overlapping tiles (a 3x3 grid, each tile half
   the photo's width and height), each tile is compared with your query, and the
   best one is outlined, with the rest of the photo dimmed. If the tiles all score
   about the same (they differ by less than 0.045), the app says no single area
   stands out instead: the match is the photo as a whole, or the thing isn't there.
   Turn on **Auto-highlight** to have this done for the top 4 photo results of every
   search (about 8 s extra). Tiles are embedded at 140 image tokens instead of the
   default 280, which halves the time (about 2 s per photo) and in testing chose an
   equally good tile overall.
7. **Re-indexing is incremental.** Every entry stores its file's path, its
   last-modified time and an `INDEX_VERSION`. When you index again, unchanged files
   are skipped, and only new or edited files are embedded (an edited file's old
   entries are replaced). If a code change alters how files are embedded, bumping
   `INDEX_VERSION` makes the next run redo every file automatically.

```
 photo ──────────► EmbeddingGemma 2 ─────────────► vector ───┐
 video ──────────► 1 frame/s ─► EmbeddingGemma 2 ─► vector ───┤
 voice note ─────► 10 s windows ► EmbeddingGemma 2 ► vectors ─┤
 PDF ────────────► 120-word passages ► EmbeddingGemma 2 ► vectors ─┼─► ChromaDB ─► nearest 12 files
 "shopping list" ► EmbeddingGemma 2 ─────────────► vector ───┘  (cosine similarity)
```

## Design choices

| Choice | Why |
|---|---|
| **EmbeddingGemma 2** | One model embeds text, images, video and audio into the same space, so you can search photos, videos and recordings with text. We load all three parts (740M parameters). It runs in bfloat16 on the Apple GPU (MPS); **float16 is avoided** because the model card warns it produces NaN embeddings. |
| **No speech-to-text** | The model understands spoken meaning directly from the audio. In testing, "vehicle repair" found a memo about a grinding noise when braking, with no shared words. That saves a second model (Whisper) of 1–3 GB, at the cost of not being able to show a transcript. |
| **PyAV for audio and video** | Decodes iPhone `.m4a` (AAC) and resamples to the 16 kHz mono the model needs, and decodes H.264 and HEVC video, including the rotation flag on portrait phone videos. It bundles its own decoders, so no ffmpeg install is required. (`librosa`, the model's suggested audio loader, is heavier and still needs ffmpeg for `.m4a`; the model's own video loader needs yet another library, `torchcodec`.) |
| **pypdf for PDFs** | Pure Python, no system libraries, and it extracts text page by page so results can say which page matched. Scanned PDFs (pictures of pages) have no text and are reported as unreadable; rendering pages as images would handle them, but needs another library. |
| **Passages with the document prompt** | Using the model card's document format scored the right passage about 0.06 higher than plain text. In testing, PDFs won their own topics ("night trains": 0.77 against 0.60 for the next file) but didn't crowd out photos and videos on visual queries (the bicycle video: 0.70 against 0.59 for the PDF). |
| **Whole-clip video embedding** | In testing, the whole clip scored about the same as its best short window or single frame, and it's the simplest option. An 8-second clip takes about 2–3 s to embed. Clips play from the start. |
| **10-second windows every 2.5 s** | Tested against 30 s pieces (off by up to 30 s), 10 s every 5 s (missed 1 of 4 topics), and 5 s every 2.5 s (equally precise). 10 s keeps more context per window. It also keeps every piece well within the model's audio limit, which the two official sources disagree on (about 5.5 vs 16 minutes). |
| **Match area on click (or top 4), not during indexing** | Doing it during indexing would make indexing about 10x slower and the database about 10x bigger. On click, you only pay for the photos you ask about; auto-highlight limits it to the top 4 photos (about 8 s per search). |
| **140 tokens per tile** | The model's image detail setting (`max_soft_tokens`) defaults to 280. At 140, tiles took 1.9 s per photo instead of 4.1 s; on 8 test cases one tile choice got worse ("tall buildings") and one got better ("a white car"). sentence-transformers has no per-call option, so the setting is changed for the duration of the tile step and restored, with a lock so indexing can't run in between. |
| **Hide unlikely matches** | Search always returns its 12 closest results, even when only one or two are relevant. Results more than 0.10 below the best match are hidden, and the page says how many. |
| **Tile spread decides whether to highlight** | Comparing the best tile with the whole photo didn't work: whole-scene queries and things not in the photo also scored higher on some tile. How much the tiles differ from *each other* did separate them: under 0.04 for "not in the photo", 0.05+ for real objects. |
| **ChromaDB** | Runs inside the app and saves to a local folder, so there's no server. The whole data layer is a handful of calls (`upsert`, `get`, `query`, `delete`), and a `type` field lets one collection hold every kind of file. LanceDB would scale better to millions of files, but that doesn't matter for a personal library. |
| **sentence-transformers** | Wraps the model in a one-line `.encode()` call and applies the recommended `SearchQuery` task prompt to queries. |
| **Streamlit** | A working UI in about 200 lines of Python, including audio and video players and the match-area highlight. |

## Requirements

- macOS with Apple Silicon (also works on CPU-only machines, just slower)
- Python 3.10+ (built and tested with 3.12)
- About 3 GB of free RAM while idle, up to about 6 GB while indexing photos
  (measured on an M3 Pro: the model's 16-bit weights take 1.49 GB on the GPU, the
  Python process 1.3 GB, and a 16-photo batch grows GPU memory to about 4.9 GB).
  Google's ~567 MB figure is for a 4/8-bit compressed build on a phone; this app
  runs the uncompressed model.
- About 1.5 GB of disk for the model download, plus about 1.5 GB for the Python packages

## Setup

```bash
# 1. Get uv (a fast Python installer), if you don't already have Python 3.10+
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment with Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install the dependencies
uv pip install -r requirements.txt
```

(With a normal Python 3.10+, `python -m venv .venv` and `pip install -r requirements.txt` work too.)

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. (The app doesn't open the browser itself:
`.streamlit/config.toml` runs it headless, which also skips Streamlit's first-run
email prompt and turns off its usage statistics.) Then:

1. Click **📁 Index a folder**, paste a folder path and click **Index folder**.
   The first launch downloads the model (about 1.5 GB, one time only).
2. Type a description in the big search box, or click one of the example searches
   under it. Use the filter buttons (**All**, **Photos**, **Voice notes**,
   **Videos**, **PDFs**) to narrow the results. Each result is a card colour-coded
   by type (blue photo, green voice note, orange video, pink PDF) with its rank and
   how strong the match is. Voice notes play from the matching moment, videos play
   in the card, and PDFs show the matching page and passage with an **Open PDF**
   button (opens in your default PDF viewer).
3. Turn on **✨ Auto-highlight** to have the matching area outlined on the top 4
   photos, or click **Show match area** on any photo.

The look (gradient header, coloured cards) comes from a small CSS block at the
top of `app.py` and the accent colours in `.streamlit/config.toml`. It uses
system fonts only, since web fonts would be downloaded from the internet. The app
follows your computer's light or dark mode.

To get voice memos off an iPhone: AirDrop them to your Mac, or drag them out of the
Voice Memos app into a folder.

To index from the terminal instead (it also prints timings):

```bash
python index_cli.py /path/to/folder
```

## Privacy: fully offline by default

- **Model:** downloaded from Hugging Face once, on the very first run. After that it
  loads from `~/.cache/huggingface` with `local_files_only=True`, so the app never
  contacts Hugging Face again.
- **App:** `.streamlit/config.toml` binds the app to `localhost`, so only this laptop
  can open it (not other devices on your Wi-Fi), and turns off Streamlit's usage
  statistics.
- **Database:** ChromaDB's anonymous telemetry is turned off in `file_search.py`.

Photos, recordings, videos, documents, embeddings and search queries never leave the machine.

## Project layout

```
file_search.py    # core logic: load model, scan folder, read audio/video/PDF, embed, store, search
app.py            # Streamlit UI
.streamlit/       # Streamlit settings (localhost only, headless, no usage statistics)
index_cli.py      # command-line indexer with timing
requirements.txt  # pinned dependency versions
file_index/       # the vector database (created on first index; delete it to start over)
```

## Limitations

- Similarity scores are relative. Use them to rank results, not as a percentage.
  In testing, good matches scored about 0.65–0.75 and unrelated files about 0.55–0.6.
  The UI labels each result by its gap to the best match: Close (within 0.05) or
  Weak (within 0.10). Results more than 0.10 behind are hidden.
- The model reads text inside images, so screenshots of text can rank highly for
  short queries that relate to their words. Descriptive queries work best.
- In **All** mode, vague queries tend to favour photos. Use the **Voice notes**
  filter when you know you're looking for a recording.
- The match-area box is coarse: it covers a quarter of the photo, not a tight
  outline. Whole-scene queries ("a city road from above") usually still get a box,
  typically around the road; it's not wrong, just less useful. A tight outline would
  need a separate object-detection model (about 600 MB more).
- Voice-note playback can start a few seconds into a topic (the window that matched
  best isn't always the one that starts exactly where the topic does).
- A video is one vector for the whole clip, so something that only appears in part
  of a clip counts for less (in testing, "snowy landscape" ranked a clip with snowy
  mountains throughout above one that turns snowy halfway).
- A video's sound is not searched; only its frames are.
- Text matches more loosely than images do, so for vague queries a PDF passage can
  rank above everything else (in testing, "shopping list" matched the street-food
  page at 0.66). The match labels help here.
- Scanned PDFs (pictures of pages) aren't searchable yet, and **Open PDF** opens the
  first page; the card tells you which page matched.
- iPhone videos recorded in HEVC are indexed fine, but may not play in Chrome's
  result card. Safari plays them.
- Voice-note search was tested with clear speech. Noisy recordings, mumbling or
  strong accents may match less reliably.
- If you move or rename files, re-index. Old entries stay in the database and
  show up as "file moved or deleted" in search results.
- Unreadable or corrupt files are reported and skipped.
