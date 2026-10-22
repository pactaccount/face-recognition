"""Pure matching decision, separate from image and camera dependencies."""


def choose_identity(names, distances, threshold=0.55):
    if len(names) != len(distances) or not names:
        raise ValueError("Expected at least one name and one distance per name")
    best = min(range(len(distances)), key=lambda index: distances[index])
    distance = float(distances[best])
    return (names[best] if distance <= threshold else "Unknown", distance)
