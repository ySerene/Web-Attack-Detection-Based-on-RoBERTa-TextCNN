import csv
import re

from utils import normalize_for_roberta

CLASS_RE = re.compile(r'^class:\s*(.+?)\s*$', re.MULTILINE)
REQUEST_LINE_RE = re.compile(
    r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(\S+)\s+HTTP/\d+(?:\.\d+)?\s*$',
    re.MULTILINE | re.IGNORECASE
)

def split_samples(text: str):
    lines = text.splitlines()
    samples = []
    current = []
    in_sample = False

    for line in lines:
        if line.startswith("Start - Id:"):
            if current:
                samples.append("\n".join(current))
                current = []
            in_sample = True
            current.append(line)
        elif line.startswith("End - Id:"):
            if in_sample:
                current.append(line)
                samples.append("\n".join(current))
                current = []
                in_sample = False
        else:
            if in_sample:
                current.append(line)

    if current:
        samples.append("\n".join(current))

    return samples

def extract_header(block: str, header_name: str) -> str:
    pattern = re.compile(
        rf'^{re.escape(header_name)}:\s*(.*?)\s*$',
        re.MULTILINE | re.IGNORECASE
    )
    m = pattern.search(block)
    return m.group(1).strip() if m else ""

def parse_one_sample(sample_text: str):
    label_match = CLASS_RE.search(sample_text)
    label = label_match.group(1).strip() if label_match else "unknown"

    if label.lower() == "valid":
        label = "normal"

    req_match = REQUEST_LINE_RE.search(sample_text)
    if req_match:
        method = req_match.group(1).strip()
        uri = req_match.group(2).strip()
    else:
        method = ""
        uri = ""

    cookie = extract_header(sample_text, "Cookie")
    referer = extract_header(sample_text, "Referer")

    lines = sample_text.splitlines()
    body = ""

    req_idx = -1
    for i, line in enumerate(lines):
        if REQUEST_LINE_RE.match(line.strip()):
            req_idx = i
            break

    if req_idx != -1:
        after_req = lines[req_idx + 1:]

        blank_idx = None
        for j, line in enumerate(after_req):
            if line.strip() == "":
                blank_idx = j
                break

        if blank_idx is not None:
            body_lines = after_req[blank_idx + 1:]
            body_lines = [x for x in body_lines if not x.startswith("End - Id:")]
            body = "\n".join(body_lines).strip()

    if body.lower() == "null":
        body = ""

    text = (
        f"[METHOD] {normalize_for_roberta(method)} "
        f"[URI] {normalize_for_roberta(uri)} "
        f"[COOKIE] {normalize_for_roberta(cookie)} "
        f"[REFERER] {normalize_for_roberta(referer)} "
        f"[BODY] {normalize_for_roberta(body)}"
    ).strip()

    text = re.sub(r"\s+", " ", text).strip()

    return {
        "label": label,
        "text": text
    }

def parse_txt_file(txt_path: str):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    samples = split_samples(content)
    rows = [parse_one_sample(sample) for sample in samples]
    print(f"processed: {txt_path}, rows={len(rows)}")
    return rows

def main():
    train_path = "data/raw/pkdd2007/xml_train.txt"
    test_path = "data/raw/pkdd2007/xml_test.txt"
    output_csv = "data/processed/ecml_pkdd_all.csv"

    rows = []
    rows.extend(parse_txt_file(train_path))
    rows.extend(parse_txt_file(test_path))

    import os
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to: {output_csv}, total={len(rows)}")

if __name__ == "__main__":
    main()