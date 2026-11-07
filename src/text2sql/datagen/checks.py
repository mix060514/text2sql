"""Layer 1: deterministic rule checks. No model is involved here."""
import re

MIN_CHARS, MAX_CHARS = 4, 80

# Simplified-only characters that commonly appear in this domain. A hit means the text is
# not Traditional Chinese. Heuristic (no opencc dependency); extend when a leak is found.
SIMPLIFIED_ONLY = set(
    "这个们说时会对发问题么为过还没进来动经现实头关应开总额销货数据询区国产单订务卖买删库"
    "统计营显图画请帮给让将从与导历钱亚欧东电软办类别户长门间两条笔吗么样预测润价"
    "层级规则写诗归无视设结构内"
)
CJK = re.compile(r"[一-鿿]")
YEAR = re.compile(r"20\d\d")


# Metric kind: amount (revenue/price), quantity (units), count (orders). Words that name the
# wrong kind, or ambiguous "sold the most" without an explicit metric, fail deterministically.
AMOUNT_WORDS = ["營收", "銷售額", "金額", "收入", "業績", "錢", "營業額", "銷售總額", "單價", "ASP", "進帳"]
QUANTITY_WORDS = ["銷量", "銷售量", "數量", "件數", "多少件", "幾件", "多少個", "幾個產品"]
COUNT_WORDS = ["訂單", "筆數", "幾單", "下單", "單子"]
AMBIGUOUS_WORDS = ["賣得最多", "賣最多", "賣得最好", "賣最好", "買最多", "買得最多", "買貨最多", "賣得多"]
FORBIDDEN = {
    "amount": QUANTITY_WORDS + ["筆數", "幾單"],
    "quantity": [w for w in AMOUNT_WORDS if w not in ("單價", "ASP")],
    "count": [w for w in AMOUNT_WORDS if w not in ("單價", "ASP")] + ["銷量", "銷售量", "件數", "多少件", "幾件"],
}
EXPLICIT = {"amount": AMOUNT_WORDS, "quantity": QUANTITY_WORDS, "count": COUNT_WORDS}


def metric_check(text: str, metric: str | None) -> list[str]:
    if metric not in FORBIDDEN:
        return []
    # "訂單數量" is an order count, not units: drop it before looking for quantity words.
    probe = text.replace("訂單數量", "訂單筆數") if metric == "count" else text
    reasons = []
    wrong = [w for w in FORBIDDEN[metric] if w in probe]
    if metric == "amount" and "訂單數量" in text:
        wrong.append("訂單數量")
    if wrong:
        reasons.append(f"metric: 指標應為{ {'amount': '金額', 'quantity': '數量', 'count': '訂單筆數'}[metric] }，卻出現 {'、'.join(dict.fromkeys(wrong))}")
    vague = [w for w in AMBIGUOUS_WORDS if w in text]
    if vague and not any(w in probe for w in EXPLICIT[metric]):
        reasons.append(f"metric: 「{vague[0]}」分不清金額或數量，需寫明指標")
    return reasons


def normalize(text: str) -> str:
    return re.sub(r"[\s，。？?！!、,.]", "", text).lower()


def rule_check(question, seed: dict, seen: set[str]) -> list[str]:
    """Return failure reasons (empty list = pass)."""
    if not isinstance(question, str) or not question.strip():
        return ["structure: 題目不是非空字串"]
    text = question.strip()
    reasons = []
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        reasons.append(f"length: {len(text)} 字，需介於 {MIN_CHARS}-{MAX_CHARS}")
    if not CJK.search(text):
        reasons.append("language: 沒有中文字")
    simplified = sorted({ch for ch in text if ch in SIMPLIFIED_ONLY})
    if simplified:
        reasons.append(f"language: 含簡體字 {''.join(simplified)}")
    lowered = text.lower()
    # Only hard entities are checked literally; concepts are judged by the label/semantic layers.
    for group in (g for g in seed.get("keep", []) if g.get("type", "entity") == "entity"):
        if not any(term.lower() in lowered for term in group["any_of"]):
            reasons.append(f"keep: 缺少「{group['name']}」（需含其一：{'、'.join(group['any_of'])}）")
    reasons += metric_check(text, seed.get("metric"))
    extra_years = set(YEAR.findall(text)) - set(YEAR.findall(seed["question"]))
    if extra_years:
        reasons.append(f"keep: 出現種子沒有的年份 {'、'.join(sorted(extra_years))}")
    key = normalize(text)
    if key == normalize(seed["question"]):
        reasons.append("duplicate: 與種子相同")
    elif key in seen:
        reasons.append("duplicate: 與其他改寫重複")
    return reasons
