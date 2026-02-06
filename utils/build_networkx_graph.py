import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from detectors.triangles import _find_centroid_tringle


def bbox_center(bbox):
    x, y, w, h = bbox
    return (int(x + w / 2), int(y + h / 2))


def build_graph_from_nodes_edges(
    image_path,
    cabins,
    triangle_centrer,
    black_cirles_points,
    hollow_cirles_points,
    edges_list,
    draw=True,
):
    """
    Costruisce un grafo NetworkX a partire da nodi ed archi rilevati,
    e opzionalmente ne visualizza la sovrapposizione sull'immagine.

    Args:
        image_path (str): Percorso dell'immagine utilizzata per la visualizzazione.
        cabins (list): Lista delle cabine rilevate con bounding box e testo associato.
        triangle_centrer (list): Lista dei triangoli rilevati.
        black_cirles_points (list): Lista dei centri dei cerchi neri pieni.
        hollow_cirles_points (list): Lista dei cerchi vuoti rilevati.
        edges_list (list): Lista degli archi [((x1, y1), (x2, y2)), ...].
        draw (bool): Se True, disegna il grafo sopra l'immagine.

    Returns:
        G (nx.Graph): Grafo costruito.
    """
    G = nx.Graph()
    # coord_to_id = {coord: i for i, coord in enumerate(node_centers)}

    # Aggiunta dei nodi regolari
    # for i, (x, y) in enumerate([n["center"] for n in node_centers]):
    #     G.add_node(i, pos=(x, y), type="node")

    G = nx.Graph()

    # Aggiunta dei nodi CABINA
    for i, cabin in enumerate(cabins):
        center = bbox_center(cabin["bbox"])

        G.add_node(
            i,
            pos=center,
            type="cabin",
            text=cabin.get("info_text") or cabin.get("cabina_id"),
            bbox=cabin["bbox"],
        )

    # Aggiunta dei nodi cerchio nero
    for x, y in black_cirles_points:
        nid = len(G.nodes)
        G.add_node(nid, pos=(x, y), type="black_circle")

    # Aggiunta dei nodi cerchio vuoto
    for p in hollow_cirles_points:
        nid = len(G.nodes)
        G.add_node(nid, pos=p["center"], type="hollow_circle")

    # Aggiunta dei centri dei triangoli
    for triangle in triangle_centrer:
        nid = len(G.nodes)
        G.add_node(
            nid, pos=tuple(map(int, _find_centroid_tringle(triangle))), type="triangle"
        )

    # Lista completa dei nodi (coordinate)
    '''
    every_node = (
        [n["center"] for n in node_centers]
        + black_cirles_points
        + [d["center"] for d in hollow_cirles_points]
        + [_find_centroid_tringle(t) for t in triangle_centrer]
    )
    '''
    every_node = (
        [bbox_center(c["bbox"]) for c in cabins]
        + black_cirles_points
        + [d["center"] for d in hollow_cirles_points]
        + [_find_centroid_tringle(t) for t in triangle_centrer]
    )

    # Creazione della mappatura tra coordinate e ID dei nodi
    coord_to_id = {coord: i for i, coord in enumerate(every_node)}

    # Associazione degli archi ai nodi più vicini
    for (x1, y1), (x2, y2) in edges_list:
        n1 = min(every_node, key=lambda c: (c[0] - x1) ** 2 + (c[1] - y1) ** 2)
        n2 = min(every_node, key=lambda c: (c[0] - x2) ** 2 + (c[1] - y2) ** 2)
        if n1 != n2:
            G.add_edge(coord_to_id[n1], coord_to_id[n2])

    if draw:
        img = cv2.imread(image_path)
        overlay = img.copy()

        # Disegno dei nodi
        # for point in node_centers:
        #     cv2.circle(overlay, point["center"], 25, (0, 0, 255), -1)

        for cabin in cabins:
            cv2.circle(
                overlay,
                bbox_center(cabin["bbox"]),
                25,
                (0, 0, 255),
                -1,
            )
            label = cabin.get("info_text") or cabin.get("cabina_id")

            if label:
                cx, cy = bbox_center(cabin["bbox"])
                cv2.putText(
                    overlay,
                    label,
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        # Disegno degli archi
        for (x1, y1), (x2, y2) in edges_list:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

        # Disegno dei cerchi neri
        for x, y in black_cirles_points:
            cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)

        # Disegno dei cerchi vuoti
        for point in hollow_cirles_points:
            cv2.circle(overlay, point["center"], 10, (0, 0, 130), -1)

        # Disegno dei centri dei triangoli
        for triangle in triangle_centrer:
            cv2.circle(
                overlay,
                tuple(map(int, _find_centroid_tringle(triangle))),
                25,
                (150, 0, 0),
                -1,
            )

        cv2.imwrite("outputs/graph_overlay.png", overlay)
        print("Visualizzazione del grafo salvata come graph_overlay.png")

        # Visualizzazione opzionale con matplotlib + NetworkX
        plt.figure(figsize=(20, 20))
        pos = nx.get_node_attributes(G, "pos")

        # Raggruppamento dei nodi per tipo
        node_types = nx.get_node_attributes(G, "type")
        node_groups = {
            "cabin": [n for n, t in node_types.items() if t == "cabin"],
            "black_circle": [n for n, t in node_types.items() if t == "black_circle"],
            "hollow_circle": [n for n, t in node_types.items() if t == "hollow_circle"],
            "triangle": [n for n, t in node_types.items() if t == "triangle"],
        }

        # Disegno degli archi
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)

        # Disegno dei nodi per tipologia
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["cabin"],
            node_shape="s",
            node_color="red",
            node_size=300,
            label="Cabina",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["black_circle"],
            node_color="black",
            node_size=200,
            label="Cerchio Nero",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["hollow_circle"],
            node_color="white",
            edgecolors="black",
            node_size=300,
            label="Cerchio Vuoto",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["triangle"],
            node_color="blue",
            node_shape="^",
            node_size=250,
            label="Centro Triangolo",
        )

        # Etichette opzionali
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="darkgreen")

        plt.legend(scatterpoints=1)
        plt.gca().invert_yaxis()
        plt.title("Grafo Estratto (NetworkX)")
        plt.axis("equal")
        plt.savefig("outputs/Extracted_Graph_NetworkX.png", bbox_inches="tight")
        plt.close()

    print(f"✅ Grafo costruito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    return G

