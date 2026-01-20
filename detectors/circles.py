import cv2
import numpy as np
from tqdm import tqdm


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

    # --- RILEVAMENTO DI CERCHI NERI PIENI ---

    # Maschera binaria dei pixel neri
    black_mask = (gray < black_thr).astype(np.uint8)
    # Immagine integrale per il calcolo rapido dell’area
    integral = cv2.integral(black_mask)

    h, w = gray.shape
    black_circle_points = []

    for x, y in tqdm(nodes_centers, desc="Detecting black circles"):
        # Bounding box della regione circolare
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w - 1, x + radius), min(h - 1, y + radius)

        # Calcolo rapido della somma nella regione rettangolare
        # utilizzando l’immagine integrale
        region_sum = (
            integral[y2 + 1, x2 + 1]
            - integral[y1, x2 + 1]
            - integral[y2 + 1, x1]
            + integral[y1, x1]
        )

        # Rapporto approssimato di pixel neri
        # (approssimazione rettangolare della regione circolare)
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
    edge_thr=50,  # Soglia per il rilevamento dei bordi
    edge_ratio_thr=0.1,  # Rapporto di pixel di bordo nell’area anulare
    inner_ratio_thr=0.5,  # Rapporto di pixel bianchi nell’area centrale (cerchio vuoto)
    ring_width=2,  # Spessore dell’anello in cui verificare la presenza dei bordi
    debug_img=False,
    output_path="outputs/04_hollow_circles.png",
    extract_text=True,
):
    """
    Rileva cerchi vuoti contenenti lettere o testo al loro interno.

    Parametri:
    -----------
    nodes_centers : list of tuples
        Lista di coordinate (x, y) da analizzare
    img : numpy array
        Immagine di input
    radius : int
        Raggio del cerchio da verificare
    edge_thr : int
        Soglia per il rilevamento dei bordi (Canny)
    edge_ratio_thr : float
        Rapporto minimo di pixel di bordo nell’area anulare
    inner_ratio_thr : float
        Rapporto minimo di pixel bianchi/luminosi nell’area centrale (per confermare il cerchio vuoto)
    ring_width : int
        Spessore dell’anello in cui verificare il bordo del cerchio
    draw : bool
        Indica se disegnare i rilevamenti sull’immagine
    extract_text : bool
        Indica se estrarre il testo tramite OCR

    Restituisce:
    -----------
    hollow_circles : list of dict
        Lista dei cerchi vuoti rilevati con le relative proprietà
    img_result : numpy array
        Immagine con i rilevamenti disegnati
    """

    img_copy = img.copy()

    # Conversione in scala di grigi se necessario
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    hollow_circles = []

    # Applicazione del rilevamento dei bordi per individuare i contorni dei cerchi
    edges = cv2.Canny(gray, edge_thr, edge_thr * 2)

    # Creazione delle immagini integrali per un calcolo rapido
    # Per i pixel bianchi/luminosi (rilevamento del centro vuoto)
    bright_mask = (gray > 200).astype(np.uint8)
    bright_integral = cv2.integral(bright_mask)

    # Per i pixel di bordo (rilevamento del contorno del cerchio)
    edge_integral = cv2.integral(edges.astype(np.uint8) // 255)

    for x, y in tqdm(nodes_centers, desc="Detecting hollow circles with letters"):
        # Definizione delle regioni
        # Cerchio esterno (cerchio completo)
        x1_outer = max(0, x - radius)
        y1_outer = max(0, y - radius)
        x2_outer = min(w - 1, x + radius)
        y2_outer = min(h - 1, y + radius)

        # Cerchio interno (centro vuoto - raggio ridotto)
        inner_radius = max(5, radius - ring_width)
        x1_inner = max(0, x - inner_radius)
        y1_inner = max(0, y - inner_radius)
        x2_inner = min(w - 1, x + inner_radius)
        y2_inner = min(h - 1, y + inner_radius)

        # Area anulare (dove dovrebbe essere presente il bordo del cerchio)
        x1_ring = max(0, x - radius)
        y1_ring = max(0, y - radius)
        x2_ring = min(w - 1, x + radius)
        y2_ring = min(h - 1, y + radius)

        # Calcolo delle aree
        inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
        outer_area = (x2_outer - x1_outer) * (y2_outer - y1_outer)
        ring_area = outer_area - inner_area
        if inner_area <= 0 or ring_area <= 0:
            continue

        # Controllo 1: l’area interna deve essere prevalentemente chiara (cerchio vuoto)
        inner_bright_sum = (
            bright_integral[y2_inner + 1, x2_inner + 1]
            - bright_integral[y1_inner, x2_inner + 1]
            - bright_integral[y2_inner + 1, x1_inner]
            + bright_integral[y1_inner, x1_inner]
        )
        inner_bright_ratio = inner_bright_sum / inner_area

        # Controllo 2: l’area anulare deve contenere un numero significativo di bordi (contorno del cerchio)
        ring_edge_sum = (
            edge_integral[y2_ring + 1, x2_ring + 1]
            - edge_integral[y1_ring, x2_ring + 1]
            - edge_integral[y2_ring + 1, x1_ring]
            + edge_integral[y1_ring, x1_ring]
        )

        # Sottrae eventuali bordi presenti nell’area interna
        inner_edge_sum = (
            edge_integral[y2_inner + 1, x2_inner + 1]
            - edge_integral[y1_inner, x2_inner + 1]
            - edge_integral[y2_inner + 1, x1_inner]
            + edge_integral[y1_inner, x1_inner]
        )

        ring_only_edges = ring_edge_sum - inner_edge_sum
        ring_edge_ratio = ring_only_edges / ring_area if ring_area > 0 else 0

        # Verifica se si tratta di un cerchio vuoto
        is_hollow = inner_bright_ratio >= inner_ratio_thr
        has_circle_edge = ring_edge_ratio >= edge_ratio_thr
        # Controllo aggiuntivo: il centro deve contenere pixel scuri (testo)
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

            # Estrazione del testo se richiesto
            # if extract_text and center_roi.size > 0:
            #     try:
            #         # Miglioramento della ROI per l’OCR
            #         roi_enhanced = cv2.resize(
            #             center_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            #         )
            #         _, roi_binary = cv2.threshold(
            #             roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            #         )

            #         # Configurazione OCR per singolo carattere
            #         text = pytesseract.image_to_string(
            #             roi_binary,
            #             config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            #         ).strip()

            #         if text:
            #             detected["text"] = text
            #     except Exception as e:
            #         detected["text"] = None

            hollow_circles.append(detected)

            # Disegno del rilevamento
            if debug_img:
                # Disegna il cerchio esterno
                cv2.circle(img_copy, (x, y), radius, (0, 255, 0), 2)
                # Disegna il punto centrale
                cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
                # Disegna il contorno del cerchio interno
                cv2.circle(img_copy, (x, y), inner_radius, (255, 255, 0), 1)

                # Disegna il testo se rilevato
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

    print(len(hollow_circles), "cerchi vuoti con lettere rilevati.")
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

    # --- RILEVAMENTO DI CERCHI NERI PIENI ---

    # Maschera binaria dei pixel neri
    black_mask = (gray < black_thr).astype(np.uint8)
    # Immagine integrale per il calcolo rapido dell’area
    integral = cv2.integral(black_mask)

    h, w = gray.shape
    black_circle_points = []

    for x, y in tqdm(nodes_centers, desc="Detecting small black circles"):
        # Bounding box della regione circolare
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w - 1, x + radius), min(h - 1, y + radius)

        # Calcolo rapido della somma nella regione rettangolare
        # utilizzando l’immagine integrale
        region_sum = (
            integral[y2 + 1, x2 + 1]
            - integral[y1, x2 + 1]
            - integral[y2 + 1, x1]
            + integral[y1, x1]
        )

        # Rapporto approssimato di pixel neri
        # (approssimazione rettangolare della regione circolare)
        black_ratio = region_sum / ((y2 - y1) * (x2 - x1))

        if black_ratio >= ratio_thr:
            black_circle_points.append((x, y))
            if debug_img:
                cv2.circle(img_copy, (x, y), radius, (0, 0, 255), 2)
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
    if debug_img:
        cv2.imwrite(output_path, img_copy)

    return black_circle_points

