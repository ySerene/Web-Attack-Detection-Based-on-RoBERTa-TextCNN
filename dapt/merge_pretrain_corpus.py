from pathlib import Path

INPUT_FILES = [
    "data/processed/dapt/owasp_pretrain.txt",
    "data/processed/dapt/pretrain_honeypot_decoded.txt",
]

OUTPUT_FILE = "data/processed/dapt/all_pretrain.txt"

seen = set()
written = 0
skipped_empty = 0
skipped_dup = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for file_path in INPUT_FILES:
        path = Path(file_path)
        if not path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        print(f"Reading: {file_path}")

        with open(path, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                line = line.strip()

                if not line:
                    skipped_empty += 1
                    continue

                if line in seen:
                    skipped_dup += 1
                    continue

                seen.add(line)
                fout.write(line + "\n")
                written += 1

print("Done.")
print(f"Output: {OUTPUT_FILE}")
print(f"Written: {written}")
print(f"Skipped empty: {skipped_empty}")
print(f"Skipped duplicate: {skipped_dup}")