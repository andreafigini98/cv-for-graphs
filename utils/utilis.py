from utils.constants import NODE_SIZE
from detectors.triangles import _find_centroid_tringle


# se un quadrato viene identificato dove c'è un triangolo, elimina il quadrato
def clean_squares(triangle_list, square_list):
    cleaned_squares = []
    half_node = NODE_SIZE // 2

    # Computazione dei centroidi del triangolo
    triangle_centers = [
        tuple(map(int, _find_centroid_tringle(t))) for t in triangle_list
    ]

    for square in square_list:
        sx, sy = square["center"]

        # Controlla se il quadrato si sovrappone con il centro di un qualsiasi triangolo
        overlaps = any(
            abs(sx - tx) < half_node and abs(sy - ty) < half_node
            for tx, ty in triangle_centers
        )

        if not overlaps:
            cleaned_squares.append(square)

    print(f"Original len {len(square_list)}, after cleaning {len(cleaned_squares)}")

    return cleaned_squares


def get_available_memory_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                kb = int(parts[1])
                return kb / (1024**2)
