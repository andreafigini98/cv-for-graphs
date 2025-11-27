import cv2
import numpy as np
from utils.constants import NODE_SIZE
from tqdm import tqdm


def _points_close(p1, p2, tol=3):
    """Return True if two points are within tol pixels."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= tol


def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    return abs(
        (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        / 2.0
    )


def _triangle_angles(p1, p2, p3):
    """Return the three internal angles of a triangle in degrees."""

    def angle(a, b, c):
        # angle at point b
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    A = angle(p2, p1, p3)
    B = angle(p1, p2, p3)
    C = angle(p1, p3, p2)
    return A, B, C


def _find_centroid_tringle(triangle):
    """
    Compute the centroid of a triangle given 3 points.
    Each point is a tuple (x, y).
    """

    p1 = triangle[0]
    p2 = triangle[1]
    p3 = triangle[2]

    x_c = int((p1[0] + p2[0] + p3[0]) / 3)
    y_c = int((p1[1] + p2[1] + p3[1]) / 3)

    return (x_c, y_c)


def _check_centroid_distance(triangle_list, new_candidate):
    """
    Returns True if the new candidate centroid is not too near to another
    triangle
    """

    # If its the first valid candidate append it
    if len(triangle_list) == 0:
        return True

    # get the centroid of all the previous candidate (can be maybe optimized)
    centroid_list = []
    for triangle in triangle_list:
        centroid_list.append(_find_centroid_tringle(triangle))
    # find the centroid of the new candidate
    c_new_candidate = _find_centroid_tringle(new_candidate)
    # return true if the new candidate is at least NODE_SIZE away from any other candidates

    for c in centroid_list:
        # If the two centroid are too close return false
        if (
            abs(c[0] - c_new_candidate[0]) < NODE_SIZE
            and abs(c[1] - c_new_candidate[1]) < NODE_SIZE
        ):
            return False
    return True


def detect_triangles_from_edges(
    img,
    img_processed,
    angle_tolerance=10,
    output_path="outputs/02_triangles.png",
    debug_img=False,
):
    """
    Detect triangles by matching two oblique edges and a base edge that share nodes.

    Args:
        img (np.ndarray): Original image (for drawing).
        edges_list (list[((x1, y1), (x2, y2))]): List of detected edges.
        angle_tolerance (float): Angle tolerance in degrees.
        length_tolerance (float): Allowable difference in edge lengths (fraction).
    """

    triangles = []

    lines = cv2.HoughLinesP(
        img_processed, 1, np.pi / 180, threshold=40, minLineLength=70, maxLineGap=10
    )
    edges_list = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            edges_list.append(((x1, y1), (x2, y2)))

    tol = 20
    # Group edges by shared endpoints
    for (x1a, y1a), (x2a, y2a) in tqdm(edges_list, desc="Detecting triangles"):
        for (x1b, y1b), (x2b, y2b) in edges_list:
            if _points_close((x1a, y1a), (x1b, y1b), tol) and not _points_close(
                (x2a, y2a), (x2b, y2b), tol
            ):
                shared = (x1a, y1a)
                other_a, other_b = (x2a, y2a), (x2b, y2b)
            elif _points_close((x2a, y2a), (x2b, y2b), tol) and not _points_close(
                (x1a, y1a), (x1b, y1b), tol
            ):
                shared = (x2a, y2a)
                other_a, other_b = (x1a, y1a), (x1b, y1b)
            elif _points_close((x1a, y1a), (x2b, y2b), tol) and not _points_close(
                (x2a, y2a), (x1b, y1b), tol
            ):
                shared = (x1a, y1a)
                other_a, other_b = (x2a, y2a), (x1b, y1b)
            elif _points_close((x2a, y2a), (x1b, y1b), tol) and not _points_close(
                (x1a, y1a), (x2b, y2b), tol
            ):
                shared = (x2a, y2a)
                other_a, other_b = (x1a, y1a), (x2b, y2b)
            else:
                continue

            # con il check dell area se ne perde uno per qualche motivo
            # area = _triangle_area(shared, other_a, other_b)
            # min_area = (NODE_SIZE * NODE_SIZE / 2) * 0.97
            # max_area = (NODE_SIZE * NODE_SIZE / 2) * 1.03

            # if not (min_area <= area <= max_area):
            #     continue  # skip spurious triangles

            desired_angles = (52.2, 63.9)  # example
            angle_tolerance = 5  # degrees

            angles = _triangle_angles(shared, other_a, other_b)
            # compare sets ignoring order
            if all(
                any(abs(a - da) <= angle_tolerance for a in angles)
                for da in desired_angles
            ):

                triangle = tuple(sorted([shared, other_a, other_b]))

                if _check_centroid_distance(
                    triangle_list=triangles, new_candidate=triangle
                ):

                    triangles.append(triangle)
                    # Draw it
                    if debug_img:
                        pts = np.array([shared, other_a, other_b], np.int32).reshape(
                            (-1, 1, 2)
                        )
                        cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    if debug_img:
        cv2.imwrite(output_path, img)
    print(f"{len(triangles)} triangles saved to {output_path}")
    return triangles
