import cv2
from utils.constants import NODE_SIZE



def fill_gaps_grid(node_centers, debug_img=False, img=None):
    # Estrae e ordina le coordinate X e Y uniche dei centri dei nodi
    xs = sorted(set([x for x, y in node_centers]))
    ys = sorted(set([y for x, y in node_centers]))

    # Trova lo step della griglia sull'asse X
    # Scarta differenze più piccole di NODE_SIZE per filtrare linee troppo ravvicinate
    x_diffs = [b - a for a, b in zip(xs, xs[1:]) if b - a > NODE_SIZE]
    x_step = min(x_diffs) if x_diffs else None

    # Trova lo step della griglia sull'asse Y
    # Scarta differenze più piccole di NODE_SIZE per filtrare linee troppo ravvicinate
    y_diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > NODE_SIZE]
    y_step = min(y_diffs) if y_diffs else None

    # Riempie i "buchi" nella griglia sull'asse X
    # Se due coordinate X consecutive hanno una distanza ~ 2 * x_step,
    # viene aggiunto il punto medio come nuova coordinata X
    tolerance = (x_step * 2) * 0.1
    for a, b in zip(xs, xs[1:]):
        diff = b - a
        if abs(diff - x_step * 2) <= tolerance:
            midpoint = (a + b) // 2
            # print(f"Doppio step rilevato in X: {a} e {b}, aggiunto: {midpoint}")
            xs.append(midpoint)

    # Riempie i "buchi" nella griglia sull'asse Y
    # Se due coordinate Y consecutive hanno una distanza ~ 2 * y_step,
    # viene aggiunto il punto medio come nuova coordinata Y
    tolerance = (y_step * 2) * 0.1
    for a, b in zip(ys, ys[1:]):
        diff = b - a
        if abs(diff - y_step * 2) <= tolerance:
            midpoint = (a + b) // 2
            ys.append(midpoint)

    # Costruisce la griglia completa come prodotto cartesiano X × Y
    grid_point = [(x, y) for x in sorted(xs) for y in sorted(ys)]

    # Visualizzazione opzionale della griglia per debug
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
    Disegna le linee della griglia solo nei punti in cui
    intersecano i centri dei nodi rilevati.

    Args:
        img_path (str): Percorso dell'immagine originale.
        node_centers (list): Lista delle coordinate dei centri nodo [(x, y), ...].
        node_size (int): Dimensione del nodo (usata per la lunghezza delle linee).
        output_path (str): Percorso del file di output.

    Returns:
        None
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {img_path}")

    # Estrae le coordinate X e Y uniche dei nodi
    xs = sorted(set([x for x, y in node_centers]))
    ys = sorted(set([y for x, y in node_centers]))

    half_size = node_size // 2
    half_size = 12000  # estensione ampia per coprire l'intera immagine

    # Disegna le linee verticali solo dove esistono nodi
    for x in xs:
        for y in ys:
            start_point = (x, y - half_size)
            end_point = (x, y + half_size)
            cv2.line(
                img, start_point, end_point, (200, 200, 200), 1, lineType=cv2.LINE_AA
            )

    # Disegna le linee orizzontali solo dove esistono nodi
    for y in ys:
        for x in xs:
            start_point = (x - half_size, y)
            end_point = (x + half_size, y)
            cv2.line(
                img, start_point, end_point, (200, 200, 200), 1, lineType=cv2.LINE_AA
            )

    # Evidenzia i centri dei nodi
    for x, y in node_centers:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)
    print(f"Griglia dei nodi salvata in {output_path}")

