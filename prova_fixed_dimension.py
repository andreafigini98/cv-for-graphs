import copy

from detectors.triangles import detect_triangles_from_edges
from detectors.squares import detect_node_grid, detect_squares_with_letters
from detectors.circles import (
    detect_black_circles,
    detect_hollow_circles_with_letters,
    detect_small_black_circles,
)
from detectors.edges import detect_edges_on_grid

from utils.constants import NODE_SIZE, CIRCLE_RADIUS
from utils.preprocessing import preprocess, remove_blue
from utils.draw_grid import draw_node_grid, fill_gaps_grid
from utils.build_networkx_graph import build_graph_from_nodes_edges
from utils.utilis import clean_squares
from utils.text_detection import detect_text
from utils.image_to_black import enhance_text


def main():

    input = "input_data/hard.jpg"

    img = preprocess(input)
    img_original = copy.deepcopy(img)
    processed_img = remove_blue(img)
    processed_img_copy = processed_img.copy()

    nodes_center = detect_node_grid(
        img, processed_img, node_size=NODE_SIZE, debug_img=True
    )
    grid_points = fill_gaps_grid(nodes_center, debug_img=True, img=input)

    triangles = detect_triangles_from_edges(
        img_original, processed_img, debug_img=True
    )

    squares = detect_squares_with_letters(
        nodes_centers=grid_points, img=img_original, debug_img=True
    )

    squares = clean_squares(triangles, squares)

    black_cirles_points = detect_black_circles(
        grid_points, img_original, debug_img=True
    )

    hollow_cirles_points = detect_hollow_circles_with_letters(
        grid_points, img, debug_img=True
    )

    edges_list = detect_edges_on_grid(input, grid_points, debug_img=True)

    build_graph_from_nodes_edges(
        input,
        squares,
        triangles,
        black_cirles_points,
        hollow_cirles_points,
        edges_list,
        draw=True,
    )

    # 🔹 Migliora contrasto testo
    #enhance_text(input, "input_data/enhanced_img.png")
    # !!! TODO: enhance_text non funziona bene con immagini jpg
    detect_text("input_data/enhanced_img.png", "outputs/annotated.png", "outputs/associations.csv")

if __name__ == "__main__":
    main()
