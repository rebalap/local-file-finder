"""
Index a folder of photos, voice notes, videos and PDFs from the command line and report how long it took.

Usage:  python index_cli.py /path/to/folder
"""

import sys
import time

import file_search

if len(sys.argv) != 2:
    sys.exit("Usage: python index_cli.py /path/to/folder")

t0 = time.perf_counter()
model = file_search.load_model()
collection = file_search.get_collection()
t1 = time.perf_counter()

stats = file_search.index_folder(
    sys.argv[1], model, collection,
    progress_callback=lambda done, total: print(f"\r  embedded {done}/{total}", end="", flush=True),
)
t2 = time.perf_counter()

print()
print(f"Model load:  {t1 - t0:.1f} s")
print(f"Indexing:    {t2 - t1:.1f} s")
print(f"Found {stats['photos']} photos, {stats['voice_notes']} voice notes, "
      f"{stats['videos']} videos and {stats['documents']} PDFs")
print(f"Indexed {stats['indexed']}, skipped {stats['skipped']}, removed {stats['removed']}, "
      f"failed {len(stats['failed'])}")
if stats["indexed"]:
    print(f"Speed:       {(t2 - t1) / stats['indexed']:.3f} s per file")
