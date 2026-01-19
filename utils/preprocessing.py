import cv2
import numpy as np
import os
import cv2
from pdf2image import convert_from_path
import numpy as np


def preprocess(image_path):
    """
    Keep only black pixels in an image, remove everything else.

    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the result.
        threshold (int): Pixel intensity threshold to consider "black". 0-255.

    Returns:
        None
    """

    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150)

    return img


def remove_blue(image, output_path="outputs/preprocessing/no_blue.png"):
    """
    Remove blue parts of an image and keep everything else.

    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the result.

    Returns:
        None
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue range in HSV
    lower_blue = np.array([100, 60, 50])  # H, S, V
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # replace blue pixel with white
    image[blue_mask > 0] = [255, 255, 255]

    cv2.imwrite(output_path, image)
    print(f"Image with blue removed saved to {output_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    finished_preprocessing = cv2.Canny(gray, 50, 150)
    # cv2.imwrite("outputs/finished_preprocessing.png", finished_preprocessing)
    return finished_preprocessing



def handle_input_file(input_path, output_dir="input_data"):
    
    input_path = os.path.abspath(input_path)

    os.makedirs(output_dir, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(input_path))
    ext = ext.lower()

    # ---------------- PDF ----------------
    if ext == ".pdf":
        pages = convert_from_path(input_path, dpi=300)

        # per semplicità usiamo solo la prima pagina
        png_path = os.path.join(output_dir, f"enhanced_img.png")
        jpg_path = os.path.join(output_dir, f"hard.jpg")

        pages[0].save(png_path, "PNG")
        pages[0].save(jpg_path, "JPEG")

        # enhance_text lavora sul PNG
        enhanced_path = os.path.join(output_dir, f"enhanced_img.png")
        enhance_text(png_path, enhanced_path)




def enhance_text(path_in, path_out, debug=False):
    """
    Rafforza il testo colorato (blu, rosso, nero) rendendolo più scuro e pieno,
    mantenendo lo sfondo bianco e migliorando contrasto e leggibilità.
    
    thickness: regola lo spessore del testo (0 = sottile, 1 = normale, 2 = più spesso)
    """
    thickness = 1

    img = cv2.imread(path_in)
    if img is None:
        raise FileNotFoundError(f"Impossibile aprire {path_in}")

    # 1️⃣ Migliora contrasto generale in LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 2️⃣ Conversione in grigio enfatizzando R e B (testi colorati)
    b, g, r = cv2.split(img_eq)
    gray = cv2.addWeighted(r, 0.45, b, 0.45, 0)
    gray = cv2.addWeighted(gray, 1.0, g, 0.15, 0)

    # 3️⃣ Migliora contrasto e luminosità
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)

    # 4️⃣ Segmenta testo scuro
    _, mask_text = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # 5️⃣ Chiudi buchi + regola spessore del testo
    ksize = 1 + thickness  # 1 = sottile, 2 = medio, 3 = più spesso
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_text = cv2.dilate(mask_text, kernel, iterations=1)

    # 6️⃣ Crea immagine con testo nero pieno su sfondo bianco
    enhanced = np.full_like(gray, 255)
    enhanced[mask_text > 0] = 0

    # 7️⃣ Sharpen leggero per contorni più nitidi
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharp)

    cv2.imwrite(path_out, enhanced)

    if debug:
        cv2.imshow("gray", gray)
        cv2.imshow("mask_text", mask_text)
        cv2.imshow("enhanced", enhanced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"✅ Immagine migliorata salvata in {path_out}")
    return enhanced


handle_input_file("image.pdf")