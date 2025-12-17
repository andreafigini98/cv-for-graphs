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
from utils.check_cabins import load_xlsx_cabins, write_csv, normalize_cabin_id, load_competenze_xlsx, assign_competenze, write_xlsx_colored

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

    associations = detect_text("input_data/enhanced_img.png", triangles, "outputs/annotated.png", "outputs/associations.csv")

    print(associations[0].keys())
    print(associations[0])

    build_graph_from_nodes_edges(
        input,
        #squares,
        associations,
        triangles,
        black_cirles_points,
        hollow_cirles_points,
        edges_list,
        draw=True,
    )
    

    cabin_set = load_xlsx_cabins("input_data/DU10-25-100834_26092025-113430.xlsx")
    print(type(cabin_set), cabin_set[:5])

    set_comp_e, set_comp_d = load_competenze_xlsx("input_data/PUNTI DI CONFINE.xlsx", debug=False)
    print(type(set_comp_e), list(set_comp_e)[:5])
    print(type(set_comp_d), list(set_comp_d)[:5])


    #new_cabin_set = normalize_cabin_id(cabin_set)
    #print(new_cabin_set)

    associations = assign_competenze(associations, set_comp_e, set_comp_d, debug = True)
    write_csv(associations, cabin_set, "outputs/associations.csv")
    write_xlsx_colored(associations, "outputs/associations_colored.xlsx")


if __name__ == "__main__":
    main()
