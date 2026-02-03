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
from utils.preprocessing import preprocess, remove_blue, handle_input_file
from utils.draw_grid import draw_node_grid, fill_gaps_grid
from utils.build_networkx_graph import build_graph_from_nodes_edges
from utils.text_detection import detect_text
from utils.check_cabins import (
    load_xlsx_cabins,
    write_csv,
    normalize_cabin_id,
    load_competenze_xlsx,
    assign_competenze,
    write_xlsx_colored,
)


def main(file_image, file_excel, debug_callback=None):

    def dbg(msg):
        if debug_callback:
            debug_callback(msg)
        else:
            print(msg)

    dbg(f"Converting input file")
    handle_input_file(file_image)
    dbg(f"Done!")

    input = "input_data/hard.jpg"

    img = preprocess(input)
    img_original = copy.deepcopy(img)
    processed_img = remove_blue(img)
    processed_img_copy = processed_img.copy()

    dbg(f"Detecting nodes")
    nodes_center = detect_node_grid(
        img, processed_img, node_size=NODE_SIZE, debug_img=True
    )
    dbg(f"Detected {len(nodes_center)} nodes")

    dbg(f"Filling gaps in grid")
    grid_points = fill_gaps_grid(nodes_center, debug_img=True, img=input)
    dbg(f"Filled gaps in grid: {len(grid_points)} points")

    dbg(f"Detecting triangles")
    triangles = detect_triangles_from_edges(img_original, processed_img, debug_img=True)
    dbg(f"Detected {len(triangles)} triangles")

    dbg(f"Detecting associations - THIS MAY TAKE A LONG TIME")
    associations = detect_text(
        "input_data/enhanced_img.png",
        triangles,
        "outputs/annotated.png",
        "outputs/associations.csv",
    )
    dbg(f"Detected associations: {len(associations)}")

    dbg(f"Detecting black circles")
    black_cirles_points = detect_black_circles(
        grid_points, img_original, debug_img=True
    )
    dbg(f"Black circles detected: {len(black_cirles_points)}")

    dbg(f"Detecting hollow black circles")
    hollow_cirles_points = detect_hollow_circles_with_letters(
        grid_points, img, debug_img=True
    )
    dbg(f"Hollow circles detected: {len(hollow_cirles_points)}")

    dbg(f"Detecting edges")
    edges_list = detect_edges_on_grid(input, associations, debug_img=True)
    dbg(f"Edges detected: {len(edges_list)}")

    dbg("Building graph")
    build_graph_from_nodes_edges(
        input,
        associations,
        triangles,
        black_cirles_points,
        hollow_cirles_points,
        edges_list,
        draw=True,
    )
    dbg("Graph built")

    dbg("Loading cabin set")
    cabin_set = load_xlsx_cabins(file_excel)
    dbg(f"Loaded cabin set ({len(cabin_set)} entries): {cabin_set[:5]}")

    set_comp_e, set_comp_d = load_competenze_xlsx(
        "input_data/PUNTI DI CONFINE.xlsx", debug=False
    )
    dbg(f"Competence sets: E={len(set_comp_e)}, D={len(set_comp_d)}")

    associations = assign_competenze(
        associations, cabin_set, set_comp_e, set_comp_d, debug=False
    )
    write_csv(associations, "outputs/associations.csv")
    write_xlsx_colored(associations, "outputs/associations_colored.xlsx")
    dbg("Excel/CSV outputs written")
