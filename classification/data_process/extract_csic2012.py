import os
import csv
import xml.etree.ElementTree as ET

from utils import normalize_for_roberta

def safe_text(x):
    return x.strip() if x else ""

def parse_headers_block(headers_text: str) -> dict:
    headers = {}
    if not headers_text:
        return headers

    for line in headers_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()

    return headers

def build_uri(path: str, query: str) -> str:
    path = safe_text(path)
    query = safe_text(query)
    return f"{path}?{query}" if query else path

def build_training_text(method: str, uri: str, headers_text: str, body: str) -> str:
    headers = parse_headers_block(headers_text)
    method = normalize_for_roberta(method)
    uri = normalize_for_roberta(uri)
    cookie = normalize_for_roberta(headers.get("cookie", ""))
    referer = normalize_for_roberta(headers.get("referer", ""))
    body = normalize_for_roberta(body)

    return f"[METHOD] {method} [URI] {uri} [COOKIE] {cookie} [REFERER] {referer} [BODY] {body}".strip()

def parse_one_xml(xml_path: str):
    rows = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sample in root.findall(".//sample"):
        method = safe_text(sample.findtext("./request/method"))
        path = safe_text(sample.findtext("./request/path"))
        query = safe_text(sample.findtext("./request/query"))
        headers_text = safe_text(sample.findtext("./request/headers"))
        body = safe_text(sample.findtext("./request/body"))
        label_type = safe_text(sample.findtext("./label/type")).lower()
        attack_name = safe_text(sample.findtext("./label/attack"))
        label = attack_name if label_type == "attack" and attack_name else "normal"
        uri = build_uri(path, query)
        text = build_training_text(method, uri, headers_text, body)
        rows.append({
            "label": label,
            "text": text
        })

    return rows

def collect_xml_rows(folder_path: str):
    all_rows = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".xml"):
            xml_path = os.path.join(folder_path, fname)
            rows = parse_one_xml(xml_path)
            all_rows.extend(rows)
            print(f"processed: {xml_path}, rows={len(rows)}")
    return all_rows

def main():
    base_dir = "data/raw/csic2012"
    attacks_dir = os.path.join(base_dir, "attacks")
    normals_dir = os.path.join(base_dir, "normals")
    output_csv = "data/processed/csic2012_train.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []
    rows.extend(collect_xml_rows(attacks_dir))
    rows.extend(collect_xml_rows(normals_dir))

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to: {output_csv}, total={len(rows)}")

if __name__ == "__main__":
    main()