import cv2
from utils.constants import NODE_SIZE


def fill_gaps_grid(node_centers, debug_img=False, img=None):
    xs = sorted(set([x for x, y in node_centers]))
    ys = sorted(set([y for x, y in node_centers]))

    # find x_step for the grid
    # remove differences smaller than NODE_SIZE to filter our very close lines
    x_diffs = [b - a for a, b in zip(xs, xs[1:]) if b - a > NODE_SIZE]
    x_step = min(x_diffs) if x_diffs else None

    # find y_step for the grid
    # remove differences smaller than NODE_SIZE to filter our very close lines
    y_diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > NODE_SIZE]
    y_step = min(y_diffs) if y_diffs else None

    # fill in the grid gaps for X
    # if there are two consecutive x coordinates with a difference of approx 2*x_step,
    # add the midpoint between them as a new x coordinate
    tolerance = (x_step * 2) * 0.1
    for a, b in zip(xs, xs[1:]):
        diff = b - a
        if abs(diff - x_step * 2) <= tolerance:
            midpoint = (a + b) // 2
            # print(f"Found double step in X: {a} and {b}, adding: {midpoint}")
            xs.append(midpoint)

    # fill in the grid gaps for Y
    # if there are two consecutive Y coordinates with a difference of approx 2*y_step,
    # add the midpoint between them as a new Y coordinate
    tolerance = (y_step * 2) * 0.1
    for a, b in zip(ys, ys[1:]):
        diff = b - a
        if abs(diff - y_step * 2) <= tolerance:
            midpoint = (a + b) // 2
            ys.append(midpoint)
    
    grid_point = [(x, y) for x in sorted(xs) for y in sorted(ys)]
    
    if debug_img:
        draw_node_grid(img, grid_point)
        
    return grid_point

def draw_node_grid(
    img_path,
    node_centers,
    node_size=40,
    output_path="outputs/05_nodes_grid.png",
):
    """
    Draw grid lines only where they intersect node centers.

    Args:
        img_path (str): Path to original image.
        node_centers (list): List of node center tuples [(x, y), ...].
        node_size (int): Size of each node (used for line length).
        output_path (str): File path to save output.

    Returns:
        None
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Extract unique x and y coordinates for node centers
    xs = sorted(set([x for x, y in node_centers]))
    ys = sorted(set([y for x, y in node_centers]))

    half_size = node_size // 2
    half_size = 12000

    # Draw vertical lines only where nodes exist
    for x in xs:
        for y in ys:
            # Draw line spanning node height
            start_point = (x, y - half_size)
            end_point = (x, y + half_size)
            cv2.line(
                img, start_point, end_point, (200, 200, 200), 1, lineType=cv2.LINE_AA
            )

    # Draw horizontal lines only where nodes exist
    for y in ys:
        for x in xs:
            start_point = (x - half_size, y)
            end_point = (x + half_size, y)
            cv2.line(
                img, start_point, end_point, (200, 200, 200), 1, lineType=cv2.LINE_AA
            )

    # mark node centers
    for x, y in node_centers:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)
    print(f"Grid lines drawn over nodes saved to {output_path}")
