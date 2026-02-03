from utils.constants import NODE_SIZE
from detectors.triangles import _find_centroid_tringle
import psutil
from pathlib import Path
import sys


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
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)


def resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and PyInstaller."""
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path
