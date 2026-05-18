import re
import html
import base64
from urllib.parse import unquote

CONTROL_CHAR_RE = re.compile(r'[\x00-\x1F\x7F\u202E]')
MULTISPACE_RE = re.compile(r'\s+')

BASE64_RE = re.compile(
    r'^(?:[A-Za-z0-9+/]{4})*'
    r'(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$'
)

UNICODE_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{4})')
HEX_ESCAPE_RE = re.compile(r'\\x([0-9a-fA-F]{2})')
PERCENT_U_RE = re.compile(r'%u([0-9a-fA-F]{4})')

def multi_unquote(text: str, max_rounds: int = 5) -> str:
    prev = None
    rounds = 0
    while text != prev and rounds < max_rounds:
        prev = text
        text = unquote(text)
        rounds += 1
    text = CONTROL_CHAR_RE.sub('', text)
    return text

def decode_percent_u(text: str) -> str:
    def repl(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)
    return PERCENT_U_RE.sub(repl, text)

def decode_slash_unicode(text: str) -> str:
    def repl_u(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)

    def repl_x(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)

    text = UNICODE_ESCAPE_RE.sub(repl_u, text)
    text = HEX_ESCAPE_RE.sub(repl_x, text)
    return text

def looks_like_base64(text: str) -> bool:
    text = text.strip()
    if len(text) < 8:
        return False
    if len(text) % 4 != 0:
        return False
    return BASE64_RE.fullmatch(text) is not None

def decode_base64_if_possible(text: str) -> str:
    try:
        decoded = base64.b64decode(text, validate=True)
        decoded_text = decoded.decode('utf-8', errors='ignore').strip()
        if not decoded_text:
            return text

        printable_ratio = sum(ch.isprintable() for ch in decoded_text) / max(len(decoded_text), 1)
        if printable_ratio < 0.85:
            return text

        return decoded_text
    except Exception:
        return text

def normalize_url(text: str) -> str:
    return re.sub(r'https?://[^/\s]+', '_URL_', text)

def normalize_for_roberta(text: str) -> str:
    # URL解码
    text = multi_unquote(text)

    # %uXXXX解码
    text = decode_percent_u(text)

    # HTML实体解码
    text = html.unescape(text)

    # \uXXXX / \xXX解码
    text = decode_slash_unicode(text)

    # Base64解码
    stripped = text.strip()
    if looks_like_base64(stripped):
        decoded = decode_base64_if_possible(stripped)
        if decoded != stripped:
            text = decoded
            text = multi_unquote(text)
            text = decode_percent_u(text)
            text = html.unescape(text)
            text = decode_slash_unicode(text)

    # 归一化
    text = text.lower()
    text = normalize_url(text)
    text = CONTROL_CHAR_RE.sub('', text)
    text = MULTISPACE_RE.sub(' ', text).strip()

    return text