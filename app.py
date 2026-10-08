"""
Streamlit UI for the local photo, voice-note, video and PDF search app.

Run with:  streamlit run app.py
"""

import os
import subprocess
import sys
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance

import file_search

st.set_page_config(page_title="File Finder", page_icon="🔎", layout="wide")

# --- Look and feel ------------------------------------------------------------
# One colour per kind of file, used for card borders and badges.
TYPE_STYLE = {
    "photo":    {"label": "Photo",      "one": "photo",      "many": "photos",      "icon": "📷", "color": "#3B82F6"},  # blue
    "voice":    {"label": "Voice note", "one": "voice note", "many": "voice notes", "icon": "🎙️", "color": "#10B981"},  # green
    "video":    {"label": "Video",      "one": "video",      "many": "videos",      "icon": "🎬", "color": "#F97316"},  # orange
    "document": {"label": "PDF",        "one": "PDF",        "many": "PDFs",        "icon": "📄", "color": "#EC4899"},  # pink
}

# Streamlit draws the page; this CSS adds the colourful touches on top. Only
# system fonts are used, because web fonts would be downloaded from the
# internet and the app promises to stay offline.
CARD_BORDERS = "\n".join(
    f'[class*="st-key-card-{kind}-"] {{ border-top: 5px solid {style["color"]} !important; }}'
    for kind, style in TYPE_STYLE.items()
)
st.html(f"""
<style>
  .hero {{
    background: linear-gradient(120deg, #7C3AED 0%, #DB2777 55%, #F59E0B 100%);
    border-radius: 22px; padding: 28px 32px; color: white; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
  }}
  .hero h1 {{ color: white; font-size: 2.6rem; font-weight: 800; margin: 0; padding: 0; }}
  .hero p {{ color: rgba(255,255,255,0.92); font-size: 1.1rem; margin: 6px 0 16px; }}
  .chip {{
    display: inline-block; background: rgba(255,255,255,0.2); border-radius: 999px;
    padding: 5px 14px; margin: 0 8px 6px 0; font-weight: 600; font-size: 0.95rem;
  }}
  /* The big search box */
  .st-key-query input {{
    font-size: 1.25rem; padding: 14px 18px; border-radius: 14px;
  }}
  .st-key-query [data-testid="stTextInputRootElement"] {{
    border: 2px solid #7C3AED; border-radius: 16px;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.15);
  }}
  /* Result cards: coloured top edge per file type, and a small lift on hover */
  [class*="st-key-card-"] {{
    border-radius: 16px !important; transition: transform .15s ease, box-shadow .15s ease;
  }}
  [class*="st-key-card-"]:hover {{
    transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,0,0,0.18);
  }}
  {CARD_BORDERS}
  .card-head {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .rank {{ font-weight: 800; font-size: 1.05rem; margin-right: 2px; }}
  .badge {{
    border-radius: 999px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700; color: white;
  }}
  .best  {{ background: linear-gradient(90deg, #7C3AED, #DB2777); }}
  .close {{ background: #059669; }}
  .weak  {{ background: #D97706; }}
</style>
""")


# st.cache_resource keeps the model and database loaded between interactions,
# so we only pay the model load once, not on every search.
@st.cache_resource(show_spinner="Loading EmbeddingGemma 2 (first time only)...")
def get_model():
    return file_search.load_model()


@st.cache_resource
def get_collection():
    return file_search.get_collection()


@st.cache_data(max_entries=200)
def make_thumbnail(path: str):
    """Small preview image. Also converts HEIC, which browsers can't display."""
    image = file_search.load_image(path)
    image.thumbnail((400, 400))
    return image


@st.cache_data(max_entries=200)
def match_area(path: str, query: str) -> dict:
    """Which part of the photo matches the query. Cached, so each is only worked out once."""
    return file_search.find_match_area(path, query, get_model())


def highlight(path: str, box: tuple) -> "Image.Image":
    """The photo with everything outside `box` dimmed and a thick red frame around it."""
    image = file_search.load_image(path)

    # Shrink first, then draw, so the frame stays thick however big the
    # original photo is. (Drawn on the full-size photo and then shrunk, the
    # frame ends up about 1 pixel wide in the results grid.)
    scale = min(400 / image.width, 400 / image.height, 1.0)
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    box = tuple(round(v * scale) for v in box)

    dimmed = ImageEnhance.Brightness(image).enhance(0.5)  # dim, but keep the rest visible
    dimmed.paste(image.crop(box), box[:2])  # put the matching area back at full brightness
    ImageDraw.Draw(dimmed).rectangle(box, outline=(255, 40, 40), width=max(6, image.width // 40))
    return dimmed


# Results scoring more than this far below the best match are hidden: they
# are the "unlikely" ones, shown only because search always returns its
# closest results, and they mostly cause confusion.
UNLIKELY_GAP = 0.10

# With auto-highlight on, this many photo results get "Show match area"
# done for them automatically (about 2 s each).
AUTO_HIGHLIGHT_COUNT = 4

# Clickable example searches under the search box. Edit these to suit your
# own photos.
EXAMPLE_SEARCHES = [
    "a white car on the road", "hot air balloons", "someone riding a bicycle",
    "sleeping on a night train", "busy traffic seen from above",
]

# Start voice notes this many seconds before the matching window, so the
# first words aren't cut off.
REWIND_SECONDS = 2

# Tell the browser what kind of audio each file is (Streamlit would label them
# all as .wav, which Safari refuses to play).
AUDIO_MIME_TYPES = {
    ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".opus": "audio/ogg",
}


def open_file(path: str) -> None:
    """Open a file in this computer's default app (e.g. a PDF in Preview).

    This works because the app runs on your own laptop: the "server" and the
    person clicking are the same machine.
    """
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.run(["xdg-open", path], check=False)


def escape_markdown(text: str) -> str:
    """Stop characters like * or _ in PDF text from being read as formatting."""
    return "".join("\\" + ch if ch in "\\`*_{}[]()#+-.!|>~$<" else ch for ch in text)


def format_time(seconds: float) -> str:
    """90 -> "1:30"."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def use_example() -> None:
    """When an example chip is clicked, put it in the search box (and un-select the chip)."""
    st.session_state.query = st.session_state.example
    st.session_state.example = None


def type_badge(result: dict) -> str:
    """Coloured badge saying what kind of file a result is (plus page or time)."""
    style = TYPE_STYLE[result["type"]]
    text = f"{style['icon']} {style['label']}"
    if result["type"] == "document":
        text += f" · p.{result['page']}"
    elif result["type"] == "voice":
        text += f" · {format_time(result['start'])}"
    return f'<span class="badge" style="background:{style["color"]}">{text}</span>'


model = get_model()
collection = get_collection()

# Photos the user clicked "Show match area" on, as (query, path) pairs.
if "highlighted" not in st.session_state:
    st.session_state.highlighted = set()

# --- Header ---------------------------------------------------------------------
counts = file_search.count_files(collection)
chips = "".join(
    f'<span class="chip">{style["icon"]} {counts[kind]} '
    f'{style["one"] if counts[kind] == 1 else style["many"]}</span>'
    for kind, style in TYPE_STYLE.items()
)
st.html(f"""
<div class="hero">
  <h1>🔎 File Finder</h1>
  <p>Describe it in plain English and find it across your photos, voice notes, videos and PDFs.
     Everything runs offline on your laptop.</p>
  {chips}
</div>
""")

# --- Search box and examples ----------------------------------------------------
query = st.text_input(
    "Search",
    key="query",
    placeholder="🔍  Describe what you're looking for…  e.g. kids playing in the snow",
    label_visibility="collapsed",
)
st.pills(
    "Try", EXAMPLE_SEARCHES, key="example", on_change=use_example, label_visibility="collapsed"
)

# --- Filters on their own row; auto-highlight and indexing below them ----------
search_in = st.segmented_control(
    "Search in",
    ["All", "📷 Photos", "🎙️ Voice notes", "🎬 Videos", "📄 PDFs"],
    default="All",
    key="search_in",
    label_visibility="collapsed",
)
only = {
    "All": None, "📷 Photos": "photo", "🎙️ Voice notes": "voice", "🎬 Videos": "video",
    "📄 PDFs": "document", None: None,  # None: the user clicked the selected option off
}[search_in]

toggle_col, index_col = st.columns([3, 1], vertical_alignment="center")
with toggle_col:
    auto_highlight = st.toggle(
        f"✨ Auto-highlight top {AUTO_HIGHLIGHT_COUNT} photos",
        key="auto_highlight",
        help="Finds and outlines the matching area on the top photo results "
        "(adds about 8 s to each search).",
    )
with index_col:
    # Indexing is something you do now and then, so it lives in a pop-up.
    with st.popover("📁 Index a folder", width="stretch"):
        folder = st.text_input("Folder path", placeholder="/Users/you/Pictures/Holiday")
        if st.button("Index folder", type="primary", width="stretch"):
            if not folder or not os.path.isdir(folder):
                st.error("That folder doesn't exist.")
            else:
                progress = st.progress(0.0, text="Scanning folder...")

                def update(done, total):
                    progress.progress(done / total, text=f"Embedding files: {done}/{total}")

                start = time.perf_counter()
                stats = file_search.index_folder(folder, model, collection, update)
                seconds = time.perf_counter() - start

                progress.progress(1.0, text="Done")
                st.success(
                    f"Found {stats['photos']} photos, {stats['voice_notes']} voice notes, "
                    f"{stats['videos']} videos and {stats['documents']} PDFs: "
                    f"indexed {stats['indexed']}, skipped {stats['skipped']} already indexed, "
                    + (f"removed {stats['removed']} no longer in the folder, " if stats["removed"] else "")
                    + f"in {seconds:.1f} s. Refresh the page to update the counts."
                )
                if stats["failed"]:
                    st.warning(f"Couldn't read {len(stats['failed'])} file(s): {stats['failed']}")

if sum(counts.values()) == 0:
    st.info("Nothing is indexed yet. Use **📁 Index a folder** to add your files.")

# --- Results ----------------------------------------------------------------------
if query:
    results = file_search.search(query, model, collection, top_k=12, only=only)

    # Hide unlikely matches: anything more than UNLIKELY_GAP below the best one.
    best_score = results[0]["score"] if results else 0
    shown = [r for r in results if best_score - r["score"] <= UNLIKELY_GAP]
    if results:
        hidden = len(results) - len(shown)
        st.caption(
            f"Showing {len(shown)} match{'es' if len(shown) != 1 else ''} for "
            f"**{escape_markdown(query)}**"
            + (f" · {hidden} unlikely match{'es' if hidden != 1 else ''} hidden" if hidden else "")
            + " · **Gap** is how far a result's score is below the best match."
        )
    results = shown

    # Photos that get their match area highlighted without a click.
    auto_paths = set()
    if auto_highlight:
        photos = [r["path"] for r in results if r["type"] == "photo"]
        auto_paths = set(photos[:AUTO_HIGHLIGHT_COUNT])

    # Show results in a 4-column grid, best match first. Each result is a card
    # with a coloured top edge for its type: rank and badges, then the photo or
    # player, then its score.
    columns = st.columns(4)
    for rank, result in enumerate(results, start=1):
        path = result["path"]
        # (The card is created inside its column: in "with A, B:" Python
        # enters A before it evaluates B.)
        with columns[(rank - 1) % 4], st.container(border=True, key=f"card-{result['type']}-{rank}"):
            # Label each result relative to the best match, since raw scores
            # only mean something compared with each other.
            gap = best_score - result["score"]
            if rank == 1:
                verdict = '<span class="badge best">★ Best match</span>'
            elif gap <= 0.05:
                verdict = '<span class="badge close">Close match</span>'
            else:
                verdict = '<span class="badge weak">Weak match</span>'
            st.html(f'<div class="card-head"><span class="rank">#{rank}</span>'
                    f'{type_badge(result)}{verdict}</div>')

            if not os.path.exists(path):
                st.write("(file moved or deleted)")
            elif result["type"] == "photo":
                show_area = (query, path) in st.session_state.highlighted or path in auto_paths
                if show_area:
                    with st.spinner("Finding the matching area (about 2 s)..."):
                        area = match_area(path, query)
                if show_area and area["stands_out"]:
                    st.image(highlight(path, area["box"]), width="stretch")
                else:
                    st.image(make_thumbnail(path), width="stretch")
            elif result["type"] == "video":
                # Videos are embedded as a whole clip, so they play from the start.
                # (Served as mp4 so Chrome plays iPhone .mov files too, as long as
                # they're H.264; HEVC .mov files only play in Safari.)
                st.video(path, format="video/mp4")
            elif result["type"] == "document":
                # PDF: show the matching passage itself, so it's clear why this
                # document came up.
                words = result["text"].split()
                preview = " ".join(words[:35]) + (" …" if len(words) > 35 else "")
                st.markdown(f"> {escape_markdown(preview)}")
                if len(words) > 35:
                    with st.expander("Full passage"):
                        st.markdown(escape_markdown(result["text"]))
                st.button(
                    "Open PDF", key=f"open-{path}", on_click=open_file, args=(path,),
                    help=f"Opens in your PDF viewer. The match is on page {result['page']}.",
                )
            else:
                # Voice note: a player that starts just before the best-matching moment.
                st.audio(
                    path,
                    format=AUDIO_MIME_TYPES[os.path.splitext(path)[1].lower()],
                    start_time=max(0, int(result["start"] - REWIND_SECONDS)),
                )

            st.caption(
                f"Score {result['score']:.3f} · Gap {gap:.3f}  \n"
                f"{escape_markdown(os.path.basename(path))}"
            )

            # "Show match area" button for photos, or what it found once clicked.
            if result["type"] == "photo" and os.path.exists(path):
                if not show_area:
                    # on_click runs before the page redraws, so the highlight
                    # appears straight away.
                    st.button(
                        "Show match area",
                        key=f"area-{path}",
                        on_click=st.session_state.highlighted.add,
                        args=((query, path),),
                    )
                elif area["stands_out"]:
                    st.caption(
                        f"🔴 Highlighted area scores {area['score']:.3f}; "
                        f"areas differ by {area['spread']:.3f}."
                    )
                else:
                    st.caption(
                        f"No single area stands out (areas differ by only "
                        f"{area['spread']:.3f}), so the match is the photo as a whole."
                    )
