import cv2
import numpy as np
import pytesseract
from tqdm import tqdm


# TODO integrarlo con detect_hollow_circles_with_letters
# che secondo me é piú robusta
def detect_black_circles(
    nodes_centers,
    img,
    radius=25,
    black_thr=50,
    ratio_thr=0.5,
    output_path="outputs/03_black_circles.png",
    debug_img=False,
):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # --- BLACK FILLED CIRCLES DETECTION ---

    # Binary mask of black pixels
    black_mask = (gray < black_thr).astype(np.uint8)
    # Integral image for fast area sum
    integral = cv2.integral(black_mask)

    h, w = gray.shape
    black_circle_points = []

    for x, y in tqdm(nodes_centers, desc="Detecting black circles"):
        # Bounding box of circular region
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w - 1, x + radius), min(h - 1, y + radius)

        # Fast rectangular region sum using the integral image
        region_sum = (
            integral[y2 + 1, x2 + 1]
            - integral[y1, x2 + 1]
            - integral[y2 + 1, x1]
            + integral[y1, x1]
        )

        # Approximate ratio (rectangular proxy)
        black_ratio = region_sum / ((y2 - y1) * (x2 - x1))

        if black_ratio >= ratio_thr:
            black_circle_points.append((x, y))
            if debug_img:
                cv2.circle(img_copy, (x, y), radius, (0, 0, 255), 2)
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
    if debug_img:
        cv2.imwrite(output_path, img_copy)

    return black_circle_points


def detect_hollow_circles_with_letters(
    nodes_centers,
    img,
    radius=25,
    edge_thr=50,  # Threshold for edge detection
    edge_ratio_thr=0.1,  # Ratio of edge pixels in ring area
    inner_ratio_thr=0.5,  # Ratio of white pixels in center (hollow)
    ring_width=2,  # Width of the ring to check for edges
    debug_img=False,
    output_path="outputs/04_hollow_circles.png",
    extract_text=True,
):
    """
    Detect hollow circles with letters/text inside.

    Parameters:
    -----------
    nodes_centers : list of tuples
        List of (x, y) coordinates to check
    img : numpy array
        Input image
    radius : int
        Radius of the circle to check
    edge_thr : int
        Threshold for edge detection (Canny)
    edge_ratio_thr : float
        Minimum ratio of edge pixels in the ring area
    inner_ratio_thr : float
        Minimum ratio of white/bright pixels in center (to confirm hollow)
    ring_width : int
        Width of the ring to check for circle edge
    draw : bool
        Whether to draw detections on image
    extract_text : bool
        Whether to extract text using OCR

    Returns:
    --------
    hollow_circles : list of dict
        List of detected hollow circles with their properties
    img_result : numpy array
        Image with drawn detections
    """

    img_copy = img.copy()

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    hollow_circles = []

    # Apply edge detection to find circle boundaries
    edges = cv2.Canny(gray, edge_thr, edge_thr * 2)

    # Create integral images for fast computation
    # For white/bright pixels (hollow center detection)
    bright_mask = (gray > 200).astype(np.uint8)
    bright_integral = cv2.integral(bright_mask)

    # For edge pixels (circle boundary detection)
    edge_integral = cv2.integral(edges.astype(np.uint8) // 255)

    for x, y in tqdm(nodes_centers, desc="Detecting hollow circles with letters"):
        # Define regions
        # Outer circle (full circle)
        x1_outer = max(0, x - radius)
        y1_outer = max(0, y - radius)
        x2_outer = min(w - 1, x + radius)
        y2_outer = min(h - 1, y + radius)

        # Inner circle (hollow center - smaller radius)
        inner_radius = max(5, radius - ring_width)
        x1_inner = max(0, x - inner_radius)
        y1_inner = max(0, y - inner_radius)
        x2_inner = min(w - 1, x + inner_radius)
        y2_inner = min(h - 1, y + inner_radius)

        # Ring area (where circle edge should be)
        x1_ring = max(0, x - radius)
        y1_ring = max(0, y - radius)
        x2_ring = min(w - 1, x + radius)
        y2_ring = min(h - 1, y + radius)

        # Calculate areas
        inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
        outer_area = (x2_outer - x1_outer) * (y2_outer - y1_outer)
        ring_area = outer_area - inner_area
        if inner_area <= 0 or ring_area <= 0:
            continue

        # Check 1: Inner area should be mostly bright/white (hollow)
        inner_bright_sum = (
            bright_integral[y2_inner + 1, x2_inner + 1]
            - bright_integral[y1_inner, x2_inner + 1]
            - bright_integral[y2_inner + 1, x1_inner]
            + bright_integral[y1_inner, x1_inner]
        )
        inner_bright_ratio = inner_bright_sum / inner_area

        # Check 2: Ring area should have significant edges (circle boundary)
        ring_edge_sum = (
            edge_integral[y2_ring + 1, x2_ring + 1]
            - edge_integral[y1_ring, x2_ring + 1]
            - edge_integral[y2_ring + 1, x1_ring]
            + edge_integral[y1_ring, x1_ring]
        )

        # Subtract inner edges if any
        inner_edge_sum = (
            edge_integral[y2_inner + 1, x2_inner + 1]
            - edge_integral[y1_inner, x2_inner + 1]
            - edge_integral[y2_inner + 1, x1_inner]
            + edge_integral[y1_inner, x1_inner]
        )

        ring_only_edges = ring_edge_sum - inner_edge_sum
        ring_edge_ratio = ring_only_edges / ring_area if ring_area > 0 else 0

        # Check if it's a hollow circle
        is_hollow = inner_bright_ratio >= inner_ratio_thr
        has_circle_edge = ring_edge_ratio >= edge_ratio_thr
        # Additional check: Center should have some dark pixels (text)
        center_roi = gray[y1_inner:y2_inner, x1_inner:x2_inner]
        has_dark_pixels = np.any(center_roi < 150) if center_roi.size > 0 else False

        # if is_hollow and has_circle_edge and has_dark_pixels:
        if is_hollow and has_dark_pixels and has_circle_edge:
            detected = {
                "center": (x, y),
                "radius": radius,
                "inner_bright_ratio": float(inner_bright_ratio),
                "ring_edge_ratio": float(ring_edge_ratio),
                "text": None,
            }

            # Extract text if requested
            # if extract_text and center_roi.size > 0:
            #     try:
            #         # Enhance ROI for better OCR
            #         roi_enhanced = cv2.resize(
            #             center_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            #         )
            #         _, roi_binary = cv2.threshold(
            #             roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            #         )

            #         # OCR configuration for single character
            #         text = pytesseract.image_to_string(
            #             roi_binary,
            #             config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            #         ).strip()

            #         if text:
            #             detected["text"] = text
            #     except Exception as e:
            #         detected["text"] = None

            hollow_circles.append(detected)

            # Draw detection
            if debug_img:
                # Draw outer circle
                cv2.circle(img_copy, (x, y), radius, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
                # Draw inner circle boundary
                cv2.circle(img_copy, (x, y), inner_radius, (255, 255, 0), 1)

                # Draw text if detected
                if detected["text"]:
                    cv2.putText(
                        img_copy,
                        detected["text"],
                        (x + radius + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

    if debug_img:
        cv2.imwrite(output_path, img_copy)

    print(len(hollow_circles), "hollow circles with letters detected.")
    return hollow_circles


def detect_small_black_circles(
    nodes_centers,
    img,
    radius=5,
    black_thr=50,
    ratio_thr=0.5,
    output_path="outputs/03.5_small_black_circles.png",
    debug_img=False,
):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # --- BLACK FILLED CIRCLES DETECTION ---

    # Binary mask of black pixels
    black_mask = (gray < black_thr).astype(np.uint8)
    # Integral image for fast area sum
    integral = cv2.integral(black_mask)

    h, w = gray.shape
    black_circle_points = []

    for x, y in tqdm(nodes_centers, desc="Detecting small black circles"):
        # Bounding box of circular region
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w - 1, x + radius), min(h - 1, y + radius)

        # Fast rectangular region sum using the integral image
        region_sum = (
            integral[y2 + 1, x2 + 1]
            - integral[y1, x2 + 1]
            - integral[y2 + 1, x1]
            + integral[y1, x1]
        )

        # Approximate ratio (rectangular proxy)
        black_ratio = region_sum / ((y2 - y1) * (x2 - x1))

        if black_ratio >= ratio_thr:
            black_circle_points.append((x, y))
            if debug_img:
                cv2.circle(img_copy, (x, y), radius, (0, 0, 255), 2)
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
    if debug_img:
        cv2.imwrite(output_path, img_copy)

    return black_circle_points
