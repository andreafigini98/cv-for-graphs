import cv2
import numpy as np
from utils.constants import NODE_SIZE, CIRCLE_RADIUS
from tqdm import tqdm
from skimage.draw import line


def detect_edges_on_grid(
    image_path,
    node_centers,
    black_threshold=50,
    tolerance=5,
    debug_img=False,
    output_path="outputs/05_edges.png",
):
    """
    Detect edges along the grid lines connecting nodes by looking for consecutive black pixels.

    Args:
        image_path (str): Path to the image.
        node_centers (list): List of node centers [(x, y), ...].
        node_size (int): Size of each node.
        black_threshold (int): Max grayscale value to consider "black".

    Returns:
        edges_list (list): List of edges as ((x1, y1), (x2, y2)) tuples.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    node_size = 50
    half_size = node_size // 2

    edges_list = []

    xs = sorted(set([x for x, y in node_centers]))
    ys = sorted(set([y for x, y in node_centers]))
    # === VERTICAL EDGES ===
    for x in xs:
        for i in range(len(ys) - 1):
            y_start, y_end = ys[i] + half_size, ys[i + 1] - half_size
            if y_end <= y_start:
                continue

            found_edge = False
            for dx in range(-tolerance, tolerance + 1):
                line_x = np.clip(x + dx, 0, gray.shape[1] - 1)
                line_pixels = gray[y_start:y_end, line_x]
                black_ratio = np.sum(line_pixels < black_threshold) / len(line_pixels)

                if black_ratio >= 0.5:
                    found_edge = True
                    break  # one good line is enough

            if found_edge:
                edges_list.append(((x, ys[i]), (x, ys[i + 1])))
                if debug_img:
                    cv2.line(
                        img,
                        (x, ys[i]),
                        (x, ys[i + 1]),
                        (0, 200, 0),
                        4,
                        lineType=cv2.LINE_AA,
                    )

    # === HORIZONTAL EDGES ===
    for y in ys:
        for i in range(len(xs) - 1):
            x_start, x_end = xs[i] + half_size, xs[i + 1] - half_size
            if x_end <= x_start:
                continue

            found_edge = False
            for dy in range(-tolerance, tolerance + 1):
                line_y = np.clip(y + dy, 0, gray.shape[0] - 1)
                line_pixels = gray[line_y, x_start:x_end]
                black_ratio = np.sum(line_pixels < black_threshold) / len(line_pixels)

                if black_ratio >= 0.5:
                    found_edge = True
                    break  # one good line is enough

            if found_edge:
                edges_list.append(((xs[i], y), (xs[i + 1], y)))
                if debug_img:
                    cv2.line(
                        img,
                        (xs[i], y),
                        (xs[i + 1], y),
                        (0, 200, 0),
                        4,
                        lineType=cv2.LINE_AA,
                    )
    # === DIAGONAL EDGES ===

    # Create a set for quick lookup
    nodes_upper_edges = [(x, y + NODE_SIZE // 2) for x, y in node_centers]
    nodes_lower_edges = [(x, y - NODE_SIZE // 2) for x, y in node_centers]
    cirle_upper_edges = [(x, y + CIRCLE_RADIUS) for x, y in node_centers]
    cirle_lower_edges = [(x, y - CIRCLE_RADIUS) for x, y in node_centers]

    nodes_edges = (
        nodes_upper_edges + nodes_lower_edges + cirle_upper_edges + cirle_lower_edges
    )

    if debug_img:
        for point in nodes_edges:
            cv2.circle(img, point, 3, (255, 0, 255), -1)

    target_slopes = [
        np.tan(np.radians(46.3)),
        np.tan(np.radians(27.6)),
        np.tan(np.radians(43.6)),
    ]
    slope_tol = 5

    # === DIAGONAL EDGES ===
    for i in tqdm(range(len(nodes_edges)), desc="Scanning diagonals"):
        x1, y1 = nodes_edges[i]

        for j in range(i + 1, len(nodes_edges)):
            x2, y2 = nodes_edges[j]
            dx, dy = x2 - x1, y2 - y1

            if dx == 0:
                continue  # skip vertical
            slope = dy / dx
            if not any(abs(abs(slope) - s) < slope_tol for s in target_slopes):
                continue  # skip lines not matching desired angles
                # Skip lines that are too long
            length = np.hypot(dx, dy)
            if length > (13 * NODE_SIZE) or length < (5 * NODE_SIZE):
                continue

            # Generate main line coordinates
            rr, cc = line(y1, x1, y2, x2)

            # Clip once
            rr = np.clip(rr, 0, gray.shape[0] - 1)
            cc = np.clip(cc, 0, gray.shape[1] - 1)

            found_edge = False
            for offset in range(-tolerance, tolerance + 1):
                rr_offset = np.clip(rr + offset, 0, gray.shape[0] - 1)
                cc_offset = np.clip(cc + offset, 0, gray.shape[1] - 1)

                line_pixels = gray[rr_offset, cc_offset]

                if np.mean(line_pixels < black_threshold) >= 0.5:
                    found_edge = True
                    break

            if found_edge:
                edges_list.append(((x1, y1), (x2, y2)))
                if debug_img:
                    cv2.line(
                        img, (x1, y1), (x2, y2), (0, 200, 0), 2, lineType=cv2.LINE_AA
                    )
    if debug_img:
        cv2.imwrite(output_path, img)

    return edges_list
