import json
import re
import html
from pathlib import Path
from urllib.parse import unquote, urlsplit
from collections import Counter

INPUT_FILE = "data/raw/dapt/honeypot/honeypot.json"
OUTPUT_FILE = "data/processed/dapt/pretrain_honeypot_decoded.txt"

# 调试输出
BAD_CASE_FILE = "data/processed/dapt/bad_cases_honeypot.txt"
MAX_BAD_CASES_TO_SAVE = 50

# 是否丢弃静态资源
DROP_STATIC = True

# 是否去重
DEDUP = True

# 最大循环解码轮数
MAX_UNQUOTE_ROUNDS = 5

# 正则
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u202E]')
MULTISPACE_RE = re.compile(r'\s+')
REQUEST_LINE_RE = re.compile(r'^([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)$')

STATIC_EXTENSIONS = {
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".map", ".bmp", ".webp"
}
STATIC_PATHS = {"/favicon.ico", "/robots.txt"}

def clean_basic(text: str) -> str:
    if not text:
        return ""
    return CONTROL_CHAR_RE.sub("", text)

def multi_unquote(text: str, max_rounds: int = MAX_UNQUOTE_ROUNDS) -> str:
    if not text:
        return ""
    prev = None
    rounds = 0
    while text != prev and rounds < max_rounds:
        prev = text
        text = unquote(text)
        rounds += 1
    return clean_basic(text)

def normalize_url_keep_path(url: str) -> str:
    url = url.strip()
    if not url or url in {"-", "null", "None"}:
        return ""
    try:
        parts = urlsplit(url)
        if parts.scheme and parts.netloc:
            path = parts.path or ""
            query = f"?{parts.query}" if parts.query else ""
            fragment = f"#{parts.fragment}" if parts.fragment else ""
            return f"_URL_{path}{query}{fragment}"
    except Exception:
        pass
    return url

def decode_and_normalize_field(text: str) -> str:
    if not text:
        return ""
    text = clean_basic(text)
    text = html.unescape(text)
    text = multi_unquote(text)
    text = normalize_url_keep_path(text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def is_static_resource(uri: str) -> bool:
    if not uri:
        return False
    path = uri.split("?", 1)[0].strip().lower()
    if path in STATIC_PATHS:
        return True
    return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)

def parse_http_request(request_raw: str):
    request_raw = clean_basic(request_raw)
    request_raw = request_raw.replace("\\r\\n", "\r\n").replace("\\n", "\n")
    request_raw = request_raw.replace("\r\n", "\n").replace("\r", "\n")

    parts = request_raw.split("\n\n", 1)
    head = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    lines = [line for line in head.split("\n") if line.strip()]
    if not lines:
        return None, "empty_request"

    request_line = lines[0].strip()
    m = REQUEST_LINE_RE.match(request_line)
    if not m:
        return None, f"bad_request_line: {request_line[:300]}"

    method, uri, version = m.groups()

    headers = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip()] = v.strip()

    return {
        "method": method,
        "uri": uri,
        "version": version,
        "headers": headers,
        "body": body.strip(),
    }, None

def extract_fields(parsed: dict) -> dict:
    headers = parsed.get("headers", {})

    method = parsed.get("method", "").strip()
    uri = decode_and_normalize_field(parsed.get("uri", "").strip())
    cookie = decode_and_normalize_field(headers.get("Cookie", "").strip())
    referer = decode_and_normalize_field(headers.get("Referer", "").strip())
    body = decode_and_normalize_field(parsed.get("body", "").strip())

    return {
        "method": method,
        "uri": uri,
        "cookie": cookie,
        "referer": referer,
        "body": body,
    }

def build_pretrain_text(fields: dict) -> str:
    return (
        f"[METHOD] {fields.get('method', '')} "
        f"[URI] {fields.get('uri', '')} "
        f"[COOKIE] {fields.get('cookie', '')} "
        f"[REFERER] {fields.get('referer', '')} "
        f"[BODY] {fields.get('body', '')}"
    ).strip()

def build_dedup_key(fields: dict) -> str:
    text = (
        f"[METHOD] {fields.get('method', '')} "
        f"[URI] {fields.get('uri', '')} "
        f"[COOKIE] {fields.get('cookie', '')} "
        f"[REFERER] {fields.get('referer', '')} "
        f"[BODY] {fields.get('body', '')}"
    )
    return MULTISPACE_RE.sub(" ", text).strip()

def process_record(line: str):
    line = line.strip()
    if not line:
        return None, None, None, "empty_line"

    try:
        outer = json.loads(line)
    except json.JSONDecodeError as e:
        return None, None, None, f"bad_outer_json: {str(e)}"

    payload_str = outer.get("payload")
    if not payload_str:
        return None, None, None, "missing_payload"

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as e:
        return None, None, None, f"bad_payload_json: {str(e)}"

    request_raw = payload.get("request_raw", "")
    if not request_raw:
        return None, None, None, "missing_request_raw"

    parsed, parse_err = parse_http_request(request_raw)
    if not parsed:
        return None, None, None, parse_err

    fields = extract_fields(parsed)

    if not fields["method"] or not fields["uri"]:
        return None, None, None, "missing_method_or_uri_after_extract"

    text = build_pretrain_text(fields)
    dedup_key = build_dedup_key(fields)
    uri = fields["uri"]

    return text, dedup_key, uri, None

def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    bad_case_path = Path(BAD_CASE_FILE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    reason_counter = Counter()
    bad_cases_saved = 0

    written = 0
    skipped_bad = 0
    skipped_dup = 0
    skipped_static = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_path.open("w", encoding="utf-8") as fout, \
         bad_case_path.open("w", encoding="utf-8") as ferr:

        for idx, line in enumerate(fin, start=1):
            text, dedup_key, uri, err = process_record(line)

            if err is not None:
                skipped_bad += 1
                reason_counter[err.split(":")[0]] += 1

                if bad_cases_saved < MAX_BAD_CASES_TO_SAVE:
                    ferr.write(f"===== BAD CASE #{bad_cases_saved + 1} =====\n")
                    ferr.write(f"LINE_NO: {idx}\n")
                    ferr.write(f"ERROR: {err}\n")
                    ferr.write("RAW_LINE:\n")
                    ferr.write(line[:5000] + "\n\n")
                    bad_cases_saved += 1
                continue

            if DROP_STATIC and is_static_resource(uri):
                skipped_static += 1
                reason_counter["static_resource"] += 1
                continue

            if DEDUP and dedup_key in seen:
                skipped_dup += 1
                reason_counter["duplicate"] += 1
                continue

            if DEDUP:
                seen.add(dedup_key)

            fout.write(text + "\n")
            written += 1

            if written % 100000 == 0:
                print(f"[INFO] written={written}, input_line={idx}")

    print("Done.")
    print(f"input_file={INPUT_FILE}")
    print(f"output_file={OUTPUT_FILE}")
    print(f"bad_case_file={BAD_CASE_FILE}")
    print(f"written={written}")
    print(f"skipped_bad={skipped_bad}")
    print(f"skipped_dup={skipped_dup}")
    print(f"skipped_static={skipped_static}")
    print("\nTop reasons:")
    for reason, cnt in reason_counter.most_common(20):
        print(f"{reason}: {cnt}")

if __name__ == "__main__":
    main()