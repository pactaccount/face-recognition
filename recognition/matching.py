import math


def choose_identity(names, distances, threshold=0.55):
    if len(names) != len(distances) or not names:
        raise ValueError("Expected at least one name and one distance per name")
    clean_distances = [
        float(d) if not (math.isnan(float(d)) or math.isinf(float(d))) else float("inf")
        for d in distances
    ]
    best = min(range(len(clean_distances)), key=lambda index: clean_distances[index])
    distance = float(clean_distances[best])
    return (names[best] if distance <= threshold else "Unknown", distance)


def compute_confidence(distance, threshold=0.55):
    """Calibrated biometric confidence score between 0.0 and 1.0."""
    if distance > threshold or math.isnan(distance) or math.isinf(distance):
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - (distance / threshold))), 4)
