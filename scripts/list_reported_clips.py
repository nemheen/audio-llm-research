"""Print archive paths of the English clips whose prompt sentence was reported.

    python scripts/list_reported_clips.py data/raw > clips.txt
    tar -xzf cv-corpus-22.0-delta-2025-06-20-en.tar.gz -C data/raw -T clips.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.data import read_tsv  # noqa: E402

RELEASE = "cv-corpus-22.0-delta-2025-06-20"

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw") / RELEASE
other = read_tsv(root, "en", "other")
reported = set(read_tsv(root, "en", "reported")["sentence_id"])
paths = sorted(set(other.loc[other["sentence_id"].isin(reported), "path"]))
sys.stdout.write("".join(f"{RELEASE}/en/clips/{p}\n" for p in paths))
