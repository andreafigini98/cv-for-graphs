import cv2
from utils.constants import NODE_SIZE
from tqdm import tqdm


def detect_node_grid(
    img,
    edges,
    output_path="outputs/01_nodes.png",
    node_min_area_ratio=0.0005,
    node_max_area_ratio=0.05,
    node_size=NODE_SIZE,
    debug_img=False,
):
    """
    Detect nodes and edges in a graph image and draw them on the image.

    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the result.
        node_min_area_ratio (float): Minimum relative area for nodes.
        node_max_area_ratio (float): Maximum relative area for nodes.

    Returns:
        None
    """

    # ---------------- Square detection ----------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    nodes = []
    nodes_centers = []

    for c in tqdm(contours, desc="Detecting square nodes"):
        area = cv2.contourArea(c)
        if not (
            node_min_area_ratio * img_area <= area <= node_max_area_ratio * img_area
        ):
            continue

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (
                abs(w - node_size) < 10 and abs(h - node_size) < 10
            ):  # allow small tolerance
                nodes.append((x, y, w, h))
                nodes_centers.append((x + w // 2, y + h // 2))
                if debug_img:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.circle(img, (x + w // 2, y + h // 2), 50, (255, 0, 0), -1)

    # Save the result
    if debug_img:
        cv2.imwrite(output_path, img)

    # deduplicate nodes_centers
    return list(set(nodes_centers))


def detect_squares_with_letters(
    nodes_centers,
    img,
    side=NODE_SIZE,
    edge_thr=80,  # Canny edge threshold
    edge_ratio_thr=0.08,  # Edge pixel ratio in border region
    inner_ratio_thr=0.5,  # Brightness ratio inside square
    border_width=4,  # Width of square border to check for edges
    shape_tolerance=0.1,  # Allowed aspect ratio deviation for square
    debug_img=False,
    output_path="outputs/04_squares_with_letters.png",
    extract_text=True,
):
    """
    Detect hollow or bright squares with letters or symbols inside.
    Includes shape verification to exclude triangles or other polygons.
    """
    import numpy as np
    import cv2
    from tqdm import tqdm
    import pytesseract

    img_copy = img.copy()

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    edges = cv2.Canny(gray, edge_thr, edge_thr * 2)

    # Integral images
    bright_mask = (gray > 200).astype(np.uint8)
    bright_integral = cv2.integral(bright_mask)
    edge_integral = cv2.integral(edges.astype(np.uint8) // 255)

    squares = []

    for x, y in tqdm(nodes_centers, desc="Detecting squares with letters"):
        half_side = side // 2

        # ROI boundaries
        x1_outer, y1_outer = max(0, x - half_side), max(0, y - half_side)
        x2_outer, y2_outer = min(w - 1, x + half_side), min(h - 1, y + half_side)

        # Skip if ROI too small
        if x2_outer - x1_outer < 10 or y2_outer - y1_outer < 10:
            continue

        # --- STEP 1: SHAPE CHECK (ensure it's square-like) ---
        roi_edges = edges[y1_outer:y2_outer, x1_outer:x2_outer]
        contours, _ = cv2.findContours(
            roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        is_square_shape = False
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            if len(approx) == 4:  # 4 vertices → candidate rectangle
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                # breakpoint()
                aspect_ratio = w_rect / float(h_rect)
                if 1 - shape_tolerance <= aspect_ratio <= 1 + shape_tolerance:
                    is_square_shape = True
                    break  # found at least one valid square
        if not is_square_shape:
            continue  # skip — not a square

        # --- STEP 2: BRIGHTNESS & EDGE RATIO CHECKS ---
        x1_inner, y1_inner = (
            max(0, x - half_side + border_width),
            max(0, y - half_side + border_width),
        )
        x2_inner, y2_inner = (
            min(w - 1, x + half_side - border_width),
            min(h - 1, y + half_side - border_width),
        )

        inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
        outer_area = (x2_outer - x1_outer) * (y2_outer - y1_outer)
        border_area = outer_area - inner_area
        if inner_area <= 0 or border_area <= 0:
            continue

        # Brightness
        inner_bright_sum = (
            bright_integral[y2_inner + 1, x2_inner + 1]
            - bright_integral[y1_inner, x2_inner + 1]
            - bright_integral[y2_inner + 1, x1_inner]
            + bright_integral[y1_inner, x1_inner]
        )
        inner_bright_ratio = inner_bright_sum / inner_area

        # Edge density
        outer_edge_sum = (
            edge_integral[y2_outer + 1, x2_outer + 1]
            - edge_integral[y1_outer, x2_outer + 1]
            - edge_integral[y2_outer + 1, x1_outer]
            + edge_integral[y1_outer, x1_outer]
        )
        inner_edge_sum = (
            edge_integral[y2_inner + 1, x2_inner + 1]
            - edge_integral[y1_inner, x2_inner + 1]
            - edge_integral[y2_inner + 1, x1_inner]
            + edge_integral[y1_inner, x1_inner]
        )
        border_edge_sum = outer_edge_sum - inner_edge_sum
        border_edge_ratio = border_edge_sum / border_area

        # --- STEP 3: Check text or dark pixels ---
        center_roi = gray[y1_inner:y2_inner, x1_inner:x2_inner]
        has_dark_pixels = np.any(center_roi < 150) if center_roi.size > 0 else False

        is_bright_inside = inner_bright_ratio >= inner_ratio_thr
        has_strong_edges = border_edge_ratio >= edge_ratio_thr

        if is_bright_inside and has_dark_pixels and has_strong_edges:
            detected = {
                "center": (x, y),
                "side": side,
                "inner_bright_ratio": float(inner_bright_ratio),
                "border_edge_ratio": float(border_edge_ratio),
                "text": None,
            }

            if extract_text and center_roi.size > 0:
                try:
                    roi_enlarged = cv2.resize(
                        center_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                    )
                    _, roi_binary = cv2.threshold(
                        roi_enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    text = pytesseract.image_to_string(
                        roi_binary,
                        config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    ).strip()
                    if text:
                        detected["text"] = text
                except Exception:
                    pass

            squares.append(detected)

            if debug_img:
                cv2.rectangle(
                    img_copy, (x1_outer, y1_outer), (x2_outer, y2_outer), (0, 255, 0), 2
                )
                cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
                if detected["text"]:
                    cv2.putText(
                        img_copy,
                        detected["text"],
                        (x + side // 2 + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

    if debug_img:
        cv2.imwrite(output_path, img_copy)

    print(len(squares), "squares with letters detected.")
    return squares


