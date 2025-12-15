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
    Build a NetworkX graph from detected nodes and edges, and optionally visualize it.

    Args:
        image_path (str): Path to the image used for visualization.
        node_centers (list): List of node coordinates [(x, y), ...].
        edges_list (list): List of edges [((x1, y1), (x2, y2)), ...].
        draw (bool): Whether to draw the graph on top of the image.

    Returns:
        G (nx.Graph): Constructed graph.
    """
    G = nx.Graph()
    # coord_to_id = {coord: i for i, coord in enumerate(node_centers)}

    # Add regular nodes
    #for i, (x, y) in enumerate([n["center"] for n in node_centers]):
    #    G.add_node(i, pos=(x, y), type="node")

    G = nx.Graph()

    # Add CABIN nodes
    for i, cabin in enumerate(cabins):
        center = bbox_center(cabin["bbox"])

        G.add_node(
            i,
            pos=center,
            type="cabin",
            text=cabin.get("info_text") or cabin.get("cabina_id"),
            bbox=cabin["bbox"],
        )


    # Add black circle nodes
    for x, y in black_cirles_points:
        nid = len(G.nodes)
        G.add_node(nid, pos=(x, y), type="black_circle")

    # Add hollow circle nodes
    for p in hollow_cirles_points:
        nid = len(G.nodes)
        G.add_node(nid, pos=p["center"], type="hollow_circle")

    # Add triangle centers
    for triangle in triangle_centrer:
        nid = len(G.nodes)
        G.add_node(
            nid, pos=tuple(map(int, _find_centroid_tringle(triangle))), type="triangle"
        )

    # breakpoint()
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


    # Create a mapping between coordinates and node IDs
    coord_to_id = {coord: i for i, coord in enumerate(every_node)}
    # Match edges to nearest nodes
    for (x1, y1), (x2, y2) in edges_list:
        n1 = min(every_node, key=lambda c: (c[0] - x1) ** 2 + (c[1] - y1) ** 2)
        n2 = min(every_node, key=lambda c: (c[0] - x2) ** 2 + (c[1] - y2) ** 2)
        if n1 != n2:
            G.add_edge(coord_to_id[n1], coord_to_id[n2])

    if draw:
        img = cv2.imread(image_path)
        overlay = img.copy()

        # Draw nodes
        #for point in node_centers:
        #    cv2.circle(overlay, point["center"], 25, (0, 0, 255), -1)

        for cabin in cabins:
            cv2.circle(
                overlay,
                #cabin["square_center"],
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



        # Draw edges
        for (x1, y1), (x2, y2) in edges_list:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

        # draw black circles
        for x, y in black_cirles_points:
            cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)

        # draw hollow circles
        for point in hollow_cirles_points:
            cv2.circle(overlay, point["center"], 10, (0, 0, 130), -1)

        # draw over triangles
        for triangle in triangle_centrer:
            cv2.circle(
                overlay,
                tuple(map(int, _find_centroid_tringle(triangle))),
                25,
                (150, 0, 0),
                -1,
            )

        cv2.imwrite("outputs/graph_overlay.png", overlay)
        print("Graph visualization saved as graph_overlay.png")

        # Optionally also visualize with matplotlib + networkx layout
        plt.figure(figsize=(20, 20))
        pos = nx.get_node_attributes(G, "pos")
        # Group nodes by type
        node_types = nx.get_node_attributes(G, "type")

        node_groups = {
            #"node": [n for n, t in node_types.items() if t == "node"],
            "cabin": [n for n, t in node_types.items() if t == "cabin"],
            "black_circle": [n for n, t in node_types.items() if t == "black_circle"],
            "hollow_circle": [n for n, t in node_types.items() if t == "hollow_circle"],
            "triangle": [n for n, t in node_types.items() if t == "triangle"],
        }

        # Draw all edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)

        # Draw each node type separately
        # node_shape in'so^>v<dph8'
        '''
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["node"],
            node_shape="s",
            node_color="red",
            node_size=200,
            label="Node",
        '''
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["cabin"],
            node_shape="s",
            node_color="red",
            node_size=300,
            label="Cabin",
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["black_circle"],
            node_color="black",
            node_size=200,
            label="Black Circle",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["hollow_circle"],
            node_color="white",
            edgecolors="black",
            node_size=300,
            label="Hollow Circle",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_groups["triangle"],
            node_color="blue",
            node_shape="^",
            node_size=250,
            label="Triangle Center",
        )

        # Optional labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="darkgreen")

        plt.legend(scatterpoints=1)
        plt.gca().invert_yaxis()
        plt.title("Extracted Graph (NetworkX)")
        plt.axis("equal")
        plt.savefig("outputs/Extracted_Graph_NetworkX.png", bbox_inches="tight")
        plt.close()

        # nx.draw(G, pos, node_color="red", edge_color="black", with_labels=True)
        # plt.gca().invert_yaxis()
        # plt.title("Extracted Graph (NetworkX)")
        # plt.savefig("outputs/Extracted Graph (NetworkX)")

    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
