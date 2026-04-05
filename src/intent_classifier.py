def classify_intent(query: str) -> str:
    q = query.lower()

    if any(word in q for word in ["best", "recommend", "suggest", "top", "most", "least", "popular", "famous", "great"]):
        return "recommendation"
    if any(word in q for word in ["bad", "complain", "worst", "dislike", "terrible", "awful", "poor", "horrible"]):
        return "complaint"
    if any(word in q for word in ["service", "price", "ambience", "food", "staff", "location", "menu", "atmosphere", "decor", "cleanliness", "parking"]):
        return "aspect"
    return "general"
