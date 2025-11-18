import cv2
import numpy as np
import easyocr
import pytesseract
import re
from typing import List, Tuple, Dict
import csv
from tqdm import tqdm
import gc
from utils.utilis import get_available_memory_gb


def easyocr_blocks(gray_img):
    """
    Esegue OCR con EasyOCR e restituisce blocchi compatibili con i successivi step.
    """
    reader = easyocr.Reader(["en", "it"], gpu=False)  # usa GPU=True se disponibile

    results = reader.readtext(
        gray_img, detail=1, paragraph=False, width_ths=0.6, ycenter_ths=0.5
    )

    text_blocks = []
    # for (bbox, text, conf) in results:
    for bbox, text, conf in tqdm(results, desc="Detecting text blocks"):
        # bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x, y = int(min(x_coords)), int(min(y_coords))
        w, h = int(max(x_coords) - x), int(max(y_coords) - y)

        if w > 8 and h > 8:
            text_blocks.append(
                {
                    "bbox": (x, y, w, h),
                    "ocr_text": text.strip(),
                    "confidence": float(conf),
                }
            )
    return text_blocks


def remove_text_inside_cabins(
    text_blocks, cabins, margin=5, center_only=True, overlap_thresh=0.3
):
    """
    Filtra i token OCR (text_blocks) rimuovendo quelli che
    - hanno il centro dentro una cabina (center_only=True), o
    - sovrappongono la cabina per > overlap_thresh della loro area.
    Restituisce la lista filtrata.
    """
    filtered = []
    for tb in text_blocks:
        x, y, w, h = tb["bbox"]
        tb_area = max(1, w * h)
        tx_center = x + w / 2.0
        ty_center = y + h / 2.0

        inside_flag = False
        for _, cbbox in cabins:
            cx, cy, cw, ch = cbbox
            # espandi cabina di margin px per sicurezza
            cx_e, cy_e = cx - margin, cy - margin
            cw_e, ch_e = cw + 2 * margin, ch + 2 * margin

            # (A) centro dentro la cabina espansa
            if center_only and (
                tx_center >= cx_e
                and tx_center <= cx_e + cw_e
                and ty_center >= cy_e
                and ty_center <= cy_e + ch_e
            ):
                inside_flag = True
                break

            # (B) percentuale di overlap tra token e cabina
            inter_x1 = max(x, cx)
            inter_x2 = min(x + w, cx + cw)
            inter_y1 = max(y, cy)
            inter_y2 = min(y + h, cy + ch)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area / tb_area > overlap_thresh:
                inside_flag = True
                break

        if not inside_flag:
            filtered.append(tb)
    return filtered


# =========================
# Consolidamento testo
# =========================

# Cosa fa:
# - Crea una maschera vuota e “riempie” i rettangoli OCR (token) come bianchi.
# - Applica un closing orizzontale con kernel largo (relativo alla larghezza immagine) per “collegare” token della stessa riga.
# - Piccola dilatazione verticale per unire spezzature minime.
# - Estrae contorni: ogni contorno ora approssima una “linea di testo” consolidata (line_boxes).
# Perché: l’unione morfologica è più robusta del semplice “gap tra token” quando i token sono frammentati o irregolari.
# Cose da tarare:
# - hor_kernel_w_frac: aumenta se vuoi unire più facilmente token orizzontali; diminuiscilo se fonde righe diverse.
# - min_area: elimina linee troppo piccole (rumore).


def merge_lines_morph(text_blocks, img_shape, hor_kernel_w_frac=0.035, min_area=80):
    """
    Unisce token sulla stessa riga via morfologia su maschera dei bbox OCR.
    Ritorna: line_boxes = [{bbox}]
    """
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in text_blocks:
        x, y, w, h = b["bbox"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    k_w = max(5, int(round(hor_kernel_w_frac * W)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)), 1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        line_boxes.append({"bbox": (x, y, w, h)})

    line_boxes.sort(key=lambda r: r["bbox"][1])
    return line_boxes


def dedup_blocks_by_iou(blocks, iou_thresh=0.6):
    """
    Deduplica blocchi (tiene i più grandi): utile dopo fusioni morfologiche.
    """

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = aw * ah + bw * bh - inter
        return inter / ua if ua > 0 else 0.0

    blocks = sorted(blocks, key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True)
    kept = []
    for r in blocks:
        if all(iou(r["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(r)
    return kept


# Cosa fa:
# - Prende i “line_boxes” (righe consolidate) e li unisce in blocchi verticali se:
#    - la distanza verticale è entro una frazione dell’altezza media di riga (max_v_gap_frac),
#    - c’è forte overlap orizzontale oppure i margini sinistri sono “vicini” (same_col),
#    - i centri x delle righe successive non divergono troppo (xcen_ok),
#    - il blocco non supera un numero massimo di righe (max_lines_per_block), tipicamente 3–4.
# - Deduplica finale dei blocchi (IoU alto) per evitare duplicati quasi identici.
# Perché: molti blocchi informativi sono composti da 3–4 righe in colonna; questa funzione li fonde in un singolo riquadro per blocco.
# Cose da tarare:
# - max_v_gap_frac e min_h_overlap_frac per righe più o meno distanziate/indentarle.
# - max_x_shift_frac se le righe non sono perfettamente allineate a sinistra.
# - max_lines_per_block per non fondere oltre 4 righe (evita unioni indesiderate).


def group_line_boxes_into_blocks(
    line_boxes,
    max_v_gap_frac,  # quanto due righe verticalmente distanti possono essere considerate parte dello stesso blocco
    min_h_overlap_frac,  # quanto devono sovrapporsi orizzontalmente (come percentuale della larghezza) per essere uniti
    max_x_shift_frac,  # quanto può cambiare la posizione orizzontale del blocco successivo.
    max_lines_per_block,
):
    """
    Unisce line_boxes contigui (stessa colonna, gap verticale moderato) in blocchi verticali.
    Ritorna: blocks [{bbox}]
    """
    if not line_boxes:
        return []

    line_boxes = sorted(line_boxes, key=lambda r: r["bbox"][1])
    avg_h = max(1, int(round(sum(b["bbox"][3] for b in line_boxes) / len(line_boxes))))
    max_v_gap = int(round(max_v_gap_frac * avg_h))
    max_w = max(b["bbox"][2] for b in line_boxes)

    def h_overlap_frac(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, bx2 = ax + aw, bx + bw
        inter = max(0, min(ax2, bx2) - max(ax, bx))
        return inter / max(1.0, min(aw, bw))

    def x_center(bb):
        return bb[0] + bb[2] / 2.0

    blocks = []
    cur = [line_boxes[0]]
    # for lb in line_boxes[1:]:
    for lb in tqdm(line_boxes[1:], desc="Grouping lines into blocks"):
        x, y, w, h = lb["bbox"]
        px, py, pw, ph = cur[-1]["bbox"]

        v_gap = y - (py + ph)
        same_col = (
            h_overlap_frac((x, y, w, h), (px, py, pw, ph)) >= min_h_overlap_frac
        ) or (abs(x - px) <= int(round(max_x_shift_frac * max_w)))
        xcen_diff = abs(x_center((x, y, w, h)) - x_center((px, py, pw, ph)))
        xcen_ok = xcen_diff <= 0.35 * max_w

        if (
            0 <= v_gap <= max_v_gap
            and same_col
            and xcen_ok
            and len(cur) < max_lines_per_block
        ):
            cur.append(lb)
        else:
            bx1 = min(b["bbox"][0] for b in cur)
            by1 = min(b["bbox"][1] for b in cur)
            bx2 = max(b["bbox"][0] + b["bbox"][2] for b in cur)
            by2 = max(b["bbox"][1] + b["bbox"][3] for b in cur)
            blocks.append({"bbox": (bx1, by1, bx2 - bx1, by2 - by1)})
            cur = [lb]

    bx1 = min(b["bbox"][0] for b in cur)
    by1 = min(b["bbox"][1] for b in cur)
    bx2 = max(b["bbox"][0] + b["bbox"][2] for b in cur)
    by2 = max(b["bbox"][1] + b["bbox"][3] for b in cur)
    blocks.append({"bbox": (bx1, by1, bx2 - bx1, by2 - by1)})

    blocks = dedup_blocks_by_iou(blocks, iou_thresh=0.7)
    return blocks


def is_inside(inner, outer):
    """Controlla se un rettangolo (inner) è completamente dentro un altro (outer)."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


# =============================
# ID interni cabina (opzionale)
# =============================

# ocr_blocks: una versione più semplice dell’OCR (psm 6); nel “slim” resta per l’estrazione dell’ID interno, dove il ROI è piccolo e ben definito.


def ocr_blocks(img: np.ndarray, lang="ita+eng") -> List[Dict]:
    """
    OCR semplice psm=6 (usato per ID interni). Meno recall rispetto a ocr_blocks_recall.
    """
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    def to_conf(v):
        try:
            return int(float(v))
        except Exception:
            return -1

    blocks = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        conf = to_conf(data["conf"][i])
        if conf <= 50 or not txt:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        blocks.append({"text": txt, "conf": conf, "bbox": (x, y, w, h)})
    return blocks


# - Ritaglia una regione interna alla cabina (con padding) per evitare i bordi.
# - Binarizza con Otsu e lancia l’OCR semplice.
# - Cerca un ID con regex tipiche (es. alfanumerici compatti).
# Cose da tarare:
# - pad (percentuale del lato): se l’ID sta vicino al bordo, riduci/incrementa.
# - regex: adattala al tuo formato ID (prefissi, sola numerica, ecc.).
# - sostituire ocr_blocks con ocr_blocks_recall per massimizzare la probabilità di lettura ID (al costo di più calcolo)


def extract_inner_id(
    img: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[str, float]:
    """
    Estrae un ID presente DENTRO la cabina (OCR sul ROI interno).
    Ritorna: (id_letto, conf) oppure ("", 0.0) se non trovato
    """
    x, y, w, h = bbox
    pad = int(0.08 * min(w, h))
    xi, yi = max(0, x + pad), max(0, y + pad)
    xe, ye = min(img.shape[1], x + w - pad), min(img.shape[0], y + h - pad)
    roi = img[yi:ye, xi:xe]
    if roi.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blocks = ocr_blocks(thr)
    id_regex = re.compile(r"^[A-Z]{0,3}\d{1,5}$|^[A-Z0-9]{1,6}$", re.I)
    best = ("", 0.0)
    for b in blocks:
        if id_regex.match(b["text"]) and b["conf"] > best[1]:
            best = (b["text"], float(b["conf"]))
    if not best[0] and blocks:
        best = max(((b["text"], float(b["conf"])) for b in blocks), key=lambda x: x[1])
    return best


# =========================
# Cabine (contour-first)
# =========================

# Cosa fa:
# - Converte in grayscale, applica CLAHE per normalizzare il contrasto a blocchi (utile con scansioni non uniformi).
# - Applica adaptive threshold invertita: i tratti scuri diventano bianchi su nero, ottimo per contorni netti.
# - Chiude piccoli gap (closing) e dilata leggermente per rinforzare contorni spezzati.
# - Estrae contorni; per ciascuno prova a semplificare in poligono (approxPolyDP) e filtra solo i quadrilateri convessi.
# - Applica filtri geometrici:
#    - Area relativa minima/massima (riduce rumore o cornici troppo grandi).
#    - Rapporto w/h (accetta quasi-quadrati).
#    - “Rettangolarità” (area contorno / area bounding box) per eliminare forme sfrangiate.
#    - Verifica “angoli quasi retti” su 4 vertici.
# - Deduplica rettangoli simili su base IoU.
# Cose da tarare:
# - min_rel_area, max_rel_area se prendi troppi/pochi rettangoli.
# - ar_min/ar_max per essere più/meno severi sulla forma quasi-quadrata.
# - rect_comp_min per aumentare/abbassare il livello di “compattezza” richiesto.


def find_squares_contours_strict(
    img: np.ndarray,
    *,
    min_rel_area: float = 0.000035,
    max_rel_area: float = 0.12,
    ar_min: float = 0.80,
    ar_max: float = 1.30,
    rect_comp_min: float = 0.88,
    right_angle_tol_deg: float = 12,
    debug_prefix: str = None,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Rileva cabine (rettangoli quasi quadrati) via contorni, CLAHE+adaptive+closing.
    Ritorna: lista di (approx4, (x,y,w,h)).
    """
    H, W = img.shape[:2]
    min_area = max(150, int(min_rel_area * W * H))
    max_area = max(200, int(max_rel_area * W * H))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr = cv2.dilate(thr, kernel, 1)
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_thr_contours.png", thr)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def has_right_angles(approx4, tol_deg=right_angle_tol_deg) -> bool:
        pts = [tuple(approx4[i][0]) for i in range(4)]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        pts.sort(key=lambda a: np.arctan2(a[1] - cy, a[0] - cx))

        def angle(a, b, c):
            ab = (a[0] - b[0], a[1] - b[1])
            cb = (c[0] - b[0], c[1] - b[1])
            na = (ab[0] ** 2 + ab[1] ** 2) ** 0.5
            nc = (cb[0] ** 2 + cb[1] ** 2) ** 0.5
            if na == 0 or nc == 0:
                return 0.0
            cosang = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / (na * nc)))
            return np.degrees(np.arccos(cosang))

        angs = [angle(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) for i in range(4)]
        return all(abs(a - 90.0) <= tol_deg for a in angs)

    candidates = []
    # for cnt in contours:
    for cnt in tqdm(contours, desc="Detecting cabins"):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in (0.02, 0.03):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < 8 or h < 8:
                continue
            ar = w / float(h)
            if not (ar_min <= ar <= ar_max):
                continue
            bbox_area = w * h
            if bbox_area <= 0:
                continue
            compact = area / float(bbox_area)
            if compact < rect_comp_min:
                continue
            if not has_right_angles(approx, tol_deg=right_angle_tol_deg):
                continue
            candidates.append((approx, (x, y, w, h)))
            break

    if not candidates:
        return []

    def iou(bb1, bb2):
        ax, ay, aw, ah = bb1
        bx, by, bw, bh = bb2
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = aw * ah + bw * bh - inter
        return inter / ua if ua > 0 else 0.0

    candidates.sort(key=lambda t: t[1][2] * t[1][3], reverse=True)
    kept = []
    for approx, bb in candidates:
        if all(iou(bb, kbb) < 0.5 for _, kbb in kept):
            kept.append((approx, bb))
    return kept


def easyocr_blocks_tiled(image, reader, tile_size=(1000, 1000), overlap=50):
    """
    Tile an image for OCR but preserve bbox structure compatible with detect_text.

    Returns a list of dicts:
        [{"bbox": [x, y, w, h], "ocr_text": str, "conf": float}, ...]
    """
    H, W = image.shape[:2]
    all_text_blocks = []

    for y0 in tqdm(range(0, H, tile_size[1] - overlap), desc="easyocr tiled"):
        y1 = min(y0 + tile_size[1], H)
        for x0 in range(0, W, tile_size[0] - overlap):
            x1 = min(x0 + tile_size[0], W)

            tile = image[y0:y1, x0:x1]

            results = reader.readtext(tile)
            for bbox, text, conf in results:
                # Compute minimal bounding rectangle for EasyOCR polygon
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                w, h = max_x - min_x, max_y - min_y

                # Adjust coordinates to original image
                all_text_blocks.append(
                    {
                        "bbox": [min_x + x0, min_y + y0, w, h],
                        "ocr_text": text.strip(),
                        "conf": conf,
                    }
                )

            del tile
            gc.collect()

    return all_text_blocks








# =============================
# Robust left-crop to remove cabin border before OCR
# =============================
def compute_cut_x_from_roi(roi, debug_path=None):
    """
    Cerca la prima colonna da sinistra dove inizia il testo (o area chiara).
    Ritorna cut_x (numero di pixel da tagliare a sinistra).
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Parametri adattivi
    scan_rows = min(max(int(0.05 * h), 8), min(60, h))  # quanti pixel in alto considerare
    max_shift = max(1, int(0.45 * w))                   # non tagliare oltre questa frazione
    min_remaining_w = max(20, int(0.10 * w))            # mantieni almeno questa larghezza

    # 1) proiezione orizzontale (media dei primi scan_rows)
    top_patch = gray[:scan_rows, :].astype(np.float32)
    col_mean = top_patch.mean(axis=0)  # shape (w,)

    # smoothing 1D
    kernel_size = 7
    k = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(col_mean, k, mode="same")

    # range dinamico e soglie
    vmin, vmax = smooth.min(), smooth.max()
    vrange = max(1e-3, vmax - vmin)
    # soglia su valore assoluto: consideriamo "chiaro" un valore abbastanza sopra il minimo
    value_thresh = vmin + 0.12 * vrange

    # 2) gradiente: cerca salto netto da scuro -> chiaro
    grad = np.diff(smooth)
    # soglia gradiente considerevole (adattiva)
    grad_thresh = max(3.0, 0.08 * vrange)

    cut_x = None

    # Preferiamo trovare un punto dove il gradiente è positivo e la smooth supera value_thresh
    candidates = np.where((np.concatenate(([0.0], grad)) > grad_thresh) & (smooth > value_thresh))[0]
    if candidates.size > 0:
        cut_x = int(max(0, candidates[0] - 1))  # piccolo left margin
    else:
        # fallback 1: primo punto dove smooth supera una soglia minima (più permissivo)
        idx = np.where(smooth > (vmin + 0.08 * vrange))[0]
        if idx.size > 0:
            cut_x = int(max(0, idx[0] - 1))

    # Fallback 2: se ancora nulla, controlla più righe e considera "colonna non scura"
    if cut_x is None:
        # consideriamo le prime few_rows e cerchiamo una colonna con non-troppi pixel scuri
        few_rows = min(max(12, scan_rows), h)
        min_dark_in_col = int(0.60 * few_rows)  # se una colonna ha >= questo, è bordo
        cut_x_found = None
        for x in range(min(max_shift, w - min_remaining_w)):
            col = gray[:few_rows, x]
            dark_count = int(np.sum(col < 100))
            if dark_count < min_dark_in_col:
                cut_x_found = x
                break
        if cut_x_found is not None:
            cut_x = int(max(0, cut_x_found - 1))

    # Safety clamps
    if cut_x is None or cut_x < 0:
        cut_x = 0
    if cut_x > max_shift:
        cut_x = max_shift
    if w - cut_x < min_remaining_w:
        # non tagliare se rimarrebbe troppo poco
        cut_x = 0

    # Debug: salva immagine con linea di taglio
    if debug_path is not None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (cut_x, 0), (cut_x, h - 1), (0, 0, 255), 1)
        cv2.imwrite(debug_path, vis)

    return cut_x







def detect_text(
    path_in: str,
    path_out: str = "annotated.png",
    csv_out: str = "associations.csv",
):
    """
    Pipeline aggiornata e coerente con le funzioni che hai nel file.
    """
    print("-------- Text Detection Phase ----------")

    img = cv2.imread(path_in)
    if img is None:
        raise FileNotFoundError(f"Impossibile aprire {path_in}")

    H, W = img.shape[:2]

    # 2) Trova cabine
    squares = find_squares_contours_strict(img)
    print(f"[DBG] Cabine rilevate: {len(squares)}")

    b, g, r = cv2.split(img)

    # 🔹 Versione per l'OCR, enfatizza i testi blu e rossi
    gray_boosted = cv2.addWeighted(b, 0.45, r, 0.45, 0)
    gray_boosted = cv2.addWeighted(gray_boosted, 1.0, g, 0.15, 0)
    gray_boosted = cv2.normalize(gray_boosted, None, 0, 255, cv2.NORM_MINMAX)

    # gray_boosted = cv2.addWeighted(b, 0.45, r, 0.45, 0)
    # gray_boosted = cv2.addWeighted(gray_boosted, 1.0, g, 0.15, 0)

    # 🔹 aumenta contrasto
    gray_boosted = cv2.normalize(gray_boosted, None, 0, 255, cv2.NORM_MINMAX)

    # 🔹 leggero sharpening per evidenziare le scritte colorate
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray_ocr = cv2.filter2D(gray_boosted, -1, kernel_sharp)

    # (Niente threshold binario qui, lo applicheremo più avanti nei singoli blocchi OCR)
    gray_ocr = gray_boosted.copy()

    # OCR globale ad alto recall
    # text_blocks = ocr_blocks_recall(gray_ocr)

    del b, g, r, gray_boosted
    gc.collect()

    # Try only if there are at least 8 gb
    if get_available_memory_gb() > 8:
        # ONE SHOT, very memory intensive
        # MAY CRASH!!!
        print(f"{get_available_memory_gb()} GB of memory, trying oneshot")
        text_blocks = easyocr_blocks(gray_ocr)
    else:
        # TILED, SLOWER BUT LESS MEMORY NEEDE
        print(f"{get_available_memory_gb()} GB of memory, going for the tiling")
        reader = easyocr.Reader(["en", "it"], gpu=False)  # CPU to save memory
        text_blocks = easyocr_blocks_tiled(
            gray_ocr, reader, tile_size=(1000, 1000), overlap=50
        )
    print(f"[DBG] Blocchi OCR trovati: {len(text_blocks)}")

    print(f"[DBG] Blocchi testuali OCR: {len(text_blocks)}")

    # rimuovi testo interno alle cabine
    text_blocks = remove_text_inside_cabins(
        text_blocks, squares, margin=8, center_only=True, overlap_thresh=0.25
    )

    # 4) Consolidamento blocchi testuali globali
    line_boxes = merge_lines_morph(
        text_blocks, (H, W), hor_kernel_w_frac=0.008, min_area=90
    )
    blocks_global = group_line_boxes_into_blocks(
        line_boxes,
        max_v_gap_frac=0.35,
        min_h_overlap_frac=0.75,
        max_x_shift_frac=0.12,
        max_lines_per_block=5,
    )

    # | Tipo di fusione indesiderata                 | Cosa modificare        | Valore consigliato |
    # | -------------------------------------------- | ---------------------- | ------------------ |
    # | Blocchi vicini orizzontalmente fusi          | `hor_kernel_w_frac` ↓  | 0.008              |
    # | Blocchi uno sopra l’altro fusi               | `max_v_gap_frac` ↓     | 0.35               |
    # | Blocchi con disallineamento orizzontale fusi | `min_h_overlap_frac` ↑ | 0.8–0.9            |

    # blocks_global = remove_blocks_overlapping_cabins(blocks_global, squares, overlap_thresh=0.30)

    print(f"[DBG] Blocchi consolidati: {len(blocks_global)}")

    for block in blocks_global:
        x, y, w, h = block["bbox"]

        inside_texts = []
        for tb in text_blocks:
            if is_inside(tb["bbox"], block["bbox"]):
                inside_texts.append(
                    (tb.get("text") or tb.get("ocr_text") or "").strip()
                )
        block["text"] = " ".join(inside_texts).strip()

    # 🔹 OCR individuale su ciascun blocco consolidato (salviamo testo dentro ogni blocco)
    # for idx, blk in enumerate(blocks_global):
    for idx, blk in tqdm(enumerate(blocks_global), desc="Single block OCR"):
        x, y, w, h = blk["bbox"]

        # se bbox invalido skip
        if w <= 0 or h <= 0:
            blk["ocr_text"] = ""
            continue

        # 🔹 1. Calcola margini con leggera riduzione a sinistra (evita bordo cabina)
        pad_top, pad_bottom, pad_right, pad_left = 1, 1, 1, 1

        # 🔹 taglio extra per evitare bordi delle cabine
        # shift_left = 1  # o 0.02 * w
        x0 = max(0, x + pad_left)
        y0 = max(0, y - pad_top)
        x1 = min(W, x + w + pad_right)
        y1 = min(H, y + h + pad_bottom)

        if x1 <= x0 or y1 <= y0:
            blk["ocr_text"] = ""
            continue

        roi = img[int(y0) : int(y1), int(x0) : int(x1)]

        
        # =============================
        #  FIX: rimuovere bordo a sinistra se il pixel (0,0) è scuro
        # =============================

        # Esegui il calcolo e taglia
        cut_x = compute_cut_x_from_roi(roi, debug_path=f"debug_cut_block_{idx:02d}.png")
        if cut_x > 0:
            roi = roi[:, cut_x:]


        roi_gray_check = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        threshold_dark = 150   # pixel sotto questo valore è considerato nero/scuro
        max_shift = int(0.20 * roi.shape[1])   # NON tagliare oltre il 20% della larghezza

        shift = 0
        while shift < max_shift:
            if roi_gray_check[0, shift] > threshold_dark:
                break
            shift += 1

        # Se serve, taglia la parte sinistra
        if shift > 0:
            roi = roi[:, shift:]



        # --------------------------------------------------------------------------------------------------------

        # --- 1. Upscaling
        roi_up = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # --- 2. Grigio + normalize + contrast
        roi_gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        roi_gray = cv2.convertScaleAbs(roi_gray, alpha=1.6, beta=-30)

        # --- 3. Threshold (testo nero su bianco)
        _, roi_bin = cv2.threshold(
            roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # --- 4. Pulizia: median blur + dilate per chiudere buchi
        roi_bin = cv2.medianBlur(roi_bin, 3)
        kernel = np.ones((2, 2), np.uint8)
        roi_bin = cv2.dilate(roi_bin, kernel, iterations=1)

        # salva ritaglio pulito per debug
        cv2.imwrite(f"img/clean_block_{idx:02d}.png", roi_bin)

        # --- 5. OCR Tesseract
        custom_config = r"--psm 6 -c preserve_interword_spaces=1 -c textord_space_size_is_variable=1"
        text = pytesseract.image_to_string(
            roi_bin, config=custom_config, lang="ita+eng"
        )

        # --- 6. Pulizia testo
        text = text.strip()
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"([A-Z])([0-9])", r"\1 \2", text)
        text = re.sub(r"([0-9])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([.,)])([A-Z0-9])", r"\1 \2", text)
        text = re.sub(r"([A-Z0-9])([(])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text).strip()

        blk["ocr_text"] = text
        # print(f"[OCR-CLEAN] Block {idx}: {repr(text)}")

    # ========== Ora assegniamo block['text'] dal 'ocr_text' ottenuto ==========
    for block in blocks_global:
        block["text"] = block.get("ocr_text", "").strip()

    # debug: disegna i blocchi globali consolidati
    ann_blocks = img.copy()
    for b in blocks_global:
        bx, by, bw, bh = b["bbox"]
        cv2.rectangle(ann_blocks, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    # 5) Disegna blocchi testuali (verde)
    annotated = img.copy()
    for b in blocks_global:
        bx, by, bw, bh = b["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    # 6) Disegna cabine (arancione) e label C{i}
    for i, (_, bbox) in enumerate(squares):
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(
            annotated,
            f"C{i}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            1,
        )

    # 7) Estrai ID interni delle cabine
    ids_per_square = []
    for _, bbox in squares:
        sid, conf = extract_inner_id(img, bbox)
        ids_per_square.append((sid.strip(), conf))

    # ===================
    # 8) ASSOCIAZIONE CABINA -> BLOCCO (con vincoli spaziali orizzontali)
    # ===================
    associations = []
    # for i, ((_, bbox), (cabina_id, conf)) in enumerate(zip(squares, ids_per_square)):
    for i, ((_, bbox), (cabina_id, conf)) in tqdm(
        enumerate(zip(squares, ids_per_square)), desc="Text-Cabins association"
    ):
        x, y, w, h = bbox

        cx, cy = x + w // 2, y + h // 2
        ref_x, ref_y = x + w, y + h  # angolo in basso a destra della cabina

        best_idx, best_dist = -1, 1e9
        best_text = ""

        for j, b in enumerate(blocks_global):
            bx, by, bw, bh = b["bbox"]
            bx_center, by_center = bx + bw // 2, by + bh // 2

            # deve essere davvero in basso a destra rispetto alla cabina
            if bx_center > ref_x and by_center > ref_y:
                dist = np.hypot(bx_center - ref_x, by_center - ref_y)
                if dist < best_dist:
                    best_idx, best_dist = j, dist
                    best_text = (b.get("text") or b.get("txt") or "").strip()

        associations.append(
            {
                "cabina_index": i,
                "cabina_id": cabina_id,
                "cabina_conf": float(conf) if conf is not None else 0.0,
                "info_text": best_text if best_idx != -1 else "",
            }
        )

        if best_idx != -1:
            bx, by, bw, bh = blocks_global[best_idx]["bbox"]
            bx_center, by_center = bx + bw // 2, by + bh // 2
            cv2.line(annotated, (cx, cy), (bx_center, by_center), (0, 0, 255), 2)

    # salva immagine e CSV come prima
    cv2.imwrite(path_out, annotated)
    print(f"✅ Annotazione salvata in {path_out}")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cabina_index", "cabina_id", "cabina_conf", "info_text"])
        for a in associations:
            writer.writerow(
                [
                    a["cabina_index"],
                    a["cabina_id"],
                    f"{a['cabina_conf']:.1f}",
                    a["info_text"].replace("\n", " ").replace("\r", "").strip(),
                ]
            )
    print(f"✅ CSV salvato in {csv_out}")
    print(
        f"🔗 Collegamenti trovati: {sum(1 for a in associations if a['info_text'])}/{len(associations)}"
    )

    results = []
    for a in associations:
        i = a["cabina_index"]
        _, bbox = squares[i]
        results.append(
            {
                "cabina_index": i,
                "bbox": bbox,
                "cabina_id": a["cabina_id"],
                "cabina_conf": a["cabina_conf"],
                "info_text": a["info_text"],
            }
        )

    return results, annotated
