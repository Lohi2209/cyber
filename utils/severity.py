SEVERITY_MAP = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "not_cyberbullying": "Safe",
    "safe": "Safe",
    "age": "Low",
    "gender": "Medium",
    "other_cyberbullying": "Medium",
    "religion": "High",
    "ethnicity": "High",
}


RULE_BASED_HIGH_RISK_KEYWORDS = (
    "worthless",
    "nobody wants you here",
    "idiot",
    "stupid",
    "ugly",
    "loser",
    "trash",
    "hate you",
    "shut up",
    "moron",
)


RULE_BASED_MEDIUM_RISK_KEYWORDS = (
    "annoying",
    "pathetic",
    "disgusting",
    "disappointing",
    "go away",
    "nobody likes you",
)


CONFIDENCE_LEVELS = {
    "Safe": 0.90,
    "Low": 0.75,
    "Medium": 0.60,
    "High": 0.85,
}


def map_category_to_severity(category: str) -> str:
    return SEVERITY_MAP.get(category, "Medium")


def _membership(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def calculate_fuzzy_severity(message: str, predicted_category: str, confidence: float) -> tuple[str, dict[str, float]]:
    normalized_message = (message or "").lower()
    words = normalized_message.split()
    high_hits = sum(keyword in normalized_message for keyword in RULE_BASED_HIGH_RISK_KEYWORDS)
    medium_hits = sum(keyword in normalized_message for keyword in RULE_BASED_MEDIUM_RISK_KEYWORDS)
    offensive_score = min(1.0, (high_hits * 0.35) + (medium_hits * 0.2))
    threat_score = min(1.0, sum(term in normalized_message for term in ("hurt", "kill", "die", "attack")) * 0.35)
    repetition_score = _membership(len(words), 8, 30) if words else 0.0
    category_score = 0.0 if predicted_category in {"not_cyberbullying", "safe"} else 0.55
    toxicity_score = max(offensive_score, category_score * confidence)
    combined_score = min(
        1.0,
        (0.35 * offensive_score)
        + (0.25 * toxicity_score)
        + (0.20 * threat_score)
        + (0.10 * repetition_score)
        + (0.10 * category_score),
    )
    if combined_score >= 0.65 or offensive_score >= 0.65 or threat_score >= 0.7:
        severity = "High"
    elif combined_score >= 0.30:
        severity = "Medium"
    elif predicted_category in {"not_cyberbullying", "safe"} and confidence >= CONFIDENCE_LEVELS["Safe"]:
        severity = "Safe"
    else:
        severity = "Low"
    return severity, {
        "offensive": round(offensive_score, 3),
        "toxicity": round(toxicity_score, 3),
        "threat": round(threat_score, 3),
        "repetition": round(repetition_score, 3),
        "combined": round(combined_score, 3),
    }


def apply_rule_based_override(message: str, predicted_category: str, severity: str) -> tuple[str, str, bool]:
    normalized_message = (message or "").lower()

    if predicted_category in {"not_cyberbullying", "safe"}:
        if any(keyword in normalized_message for keyword in RULE_BASED_HIGH_RISK_KEYWORDS):
            return "other_cyberbullying", "High", True

        if any(keyword in normalized_message for keyword in RULE_BASED_MEDIUM_RISK_KEYWORDS):
            return "other_cyberbullying", "Medium", True

    return predicted_category, severity, False


def build_suggestion(severity: str) -> str:
    if severity == "Safe":
        return "No harmful content detected."
    if severity == "Low":
        return "Use respectful language and avoid insensitive terms."
    if severity == "Medium":
        return "This content may be offensive. Review or edit before posting."
    if severity == "High":
        return "Escalate for moderator or administrator review."
    return "Review the message carefully."
