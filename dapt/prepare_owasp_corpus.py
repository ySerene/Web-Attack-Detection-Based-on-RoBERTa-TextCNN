import re
from pathlib import Path

INPUT_ROOT = "data/raw/dapt/owasp"
OUTPUT_FILE = "data/processed/dapt/owasp_pretrain.txt"

# 是否去重
DEDUP = True

# 基础正则
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u202E]')
MULTISPACE_RE = re.compile(r'\s+')
REQUEST_LINE_RE = re.compile(r'^([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)$')
SECTION_RE = re.compile(r'--([0-9a-fA-F]+)-([A-Z])--\n', re.MULTILINE)

def clean_basic(text: str) -> str:
    if not text:
        return ""
    return CONTROL_CHAR_RE.sub("", text)

def normalize_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()

def parse_audit_log_text(text: str):
    """
    把一个ModSecurity audit log文件切成多个事务：
    --txid-A--
    --txid-B--
    --txid-C--
    ...
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return []

    transactions = []
    current_txid = None
    current_sections = {}

    for i, m in enumerate(matches):
        txid = m.group(1)
        section = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip("\n")

        if current_txid is None:
            current_txid = txid

        if txid != current_txid:
            transactions.append((current_txid, current_sections))
            current_txid = txid
            current_sections = {}

        current_sections[section] = content

    if current_sections:
        transactions.append((current_txid, current_sections))

    return transactions

def parse_request_section_b(section_b: str):
    """
    解析B段：
    POST /xmlrpc.php HTTP/1.1
    Host: ...
    Cookie: ...
    Referer: ...
    User-Agent: ...
    """
    lines = [line for line in section_b.splitlines() if line.strip()]
    if not lines:
        return None

    request_line = lines[0].strip()
    m = REQUEST_LINE_RE.match(request_line)
    if not m:
        return None

    method, uri, version = m.groups()
    headers = {}

    for line in lines[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()

    return {
        "method": method,
        "uri": uri,
        "version": version,
        "headers": headers,
    }

def build_pretrain_text(req: dict, body: str):
    """
    最终输出格式：不包含USER_AGENT
    """
    headers = req.get("headers", {})

    method = clean_basic(req.get("method", "").strip())
    uri = clean_basic(req.get("uri", "").strip())
    cookie = clean_basic(headers.get("cookie", "").strip())
    referer = clean_basic(headers.get("referer", "").strip())
    body = clean_basic(body.strip())

    parts = [
        f"[METHOD] {method}",
        f"[URI] {uri}",
        f"[COOKIE] {cookie}",
        f"[REFERER] {referer}",
        f"[BODY] {body}",
    ]
    return normalize_spaces(" ".join(parts))

def build_dedup_key(req: dict, body: str):
    """
    去重键：额外加入USER_AGENT
    这样路径相同但客户端不同的请求不会被全去掉
    """
    headers = req.get("headers", {})

    method = clean_basic(req.get("method", "").strip())
    uri = clean_basic(req.get("uri", "").strip())
    cookie = clean_basic(headers.get("cookie", "").strip())
    referer = clean_basic(headers.get("referer", "").strip())
    user_agent = clean_basic(headers.get("user-agent", "").strip())
    body = clean_basic(body.strip())

    key = (
        f"[METHOD] {method} "
        f"[URI] {uri} "
        f"[COOKIE] {cookie} "
        f"[REFERER] {referer} "
        f"[USER_AGENT] {user_agent} "
        f"[BODY] {body}"
    )
    return normalize_spaces(key)

def process_file(file_path: Path):
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return [], {"read_error": 1}

    txs = parse_audit_log_text(text)
    results = []
    stats = {
        "ok": 0,
        "missing_B": 0,
        "bad_request_section": 0,
        "empty_sample": 0,
        "no_transactions": 0,
        "read_error": 0,
    }

    if not txs:
        stats["no_transactions"] += 1
        return results, stats

    for txid, sections in txs:
        section_b = sections.get("B", "")
        section_c = sections.get("C", "")

        if not section_b:
            stats["missing_B"] += 1
            continue

        req = parse_request_section_b(section_b)
        if not req:
            stats["bad_request_section"] += 1
            continue

        sample = build_pretrain_text(req=req, body=section_c)
        dedup_key = build_dedup_key(req=req, body=section_c)

        if not sample or sample == "[METHOD] [URI] [COOKIE] [REFERER] [BODY]":
            stats["empty_sample"] += 1
            continue

        results.append((sample, dedup_key))
        stats["ok"] += 1

    return results, stats

def main():
    input_root = Path(INPUT_ROOT)
    output_file = Path(OUTPUT_FILE)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in input_root.rglob("*") if p.is_file()]
    print(f"Found files: {len(all_files)}")

    seen = set()
    written = 0
    total_stats = {
        "ok": 0,
        "missing_B": 0,
        "bad_request_section": 0,
        "empty_sample": 0,
        "no_transactions": 0,
        "duplicate": 0,
        "read_error": 0,
    }

    with output_file.open("w", encoding="utf-8") as fout:
        for idx, file_path in enumerate(all_files, start=1):
            samples, stats = process_file(file_path)

            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

            for sample, dedup_key in samples:
                if DEDUP and dedup_key in seen:
                    total_stats["duplicate"] += 1
                    continue

                if DEDUP:
                    seen.add(dedup_key)

                fout.write(sample + "\n")
                written += 1

            if idx % 100 == 0:
                print(f"[INFO] processed_files={idx}/{len(all_files)} written={written}")

    print("Done.")
    print(f"output_file={OUTPUT_FILE}")
    print(f"written={written}")
    print("Stats:")
    for k, v in total_stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()