import re
import csv
from openpyxl import load_workbook
import math


'''
def load_xlsx_cabins(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None, dtype=str)

    # seconda colonna, da riga 10, solo righe pari
    raw_codes = df.iloc[9:, 1].iloc[::2]

    cabin_set = {
        normalize_cabin_id(code)
        for code in raw_codes
        if normalize_cabin_id(code)
    }

    return cabin_set
'''


def load_xlsx_cabins(path_xlsx):
    wb = load_workbook(path_xlsx, data_only=True)
    ws = wb.active

    cabins = []

    # seconda colonna = B
    # righe pari a partire dalla 10
    for row in range(10, ws.max_row + 1, 2):
        val = ws.cell(row=row, column=2).value

        if val is None:
            continue

        # scarta NaN
        if isinstance(val, float) and math.isnan(val):
            continue

        cabins.append(str(val).strip())

    return cabins








def normalize_TUG(s):
    if not s:
        return ""

    # 1. togli gli spazi tra lettera e numero
    s = re.sub(r"([TUG])\s+", r"\1", s)

    # 2. se c'è una O dopo la lettera → è uno 0
    s = re.sub(r"([TUG])O(\d)", lambda m: m.group(1) + "0" + m.group(2), s)

    # 3. TO 1 → T01
    s = re.sub(r"([TUG])\s*O\s*(\d)", lambda m: m.group(1) + "0" + m.group(2), s)

    # 4. spazi residui
    return s.strip()




def parse_info_text(s):
    original = s.strip()

    # --- 1) Cabina ID generale (AA 10-2-123456)
    cabina_pattern = r"([A-Z]{2}\s*\d{1,2}-\d-\d{6})"
    cabina_match = re.search(cabina_pattern, original)
    cabina_id = cabina_match.group(1).strip() if cabina_match else ""

    rest = original[len(cabina_id):].strip() if cabina_id else original

    # --- 2) Estrarre tutti i T/U/G (anche multipli)
    #TUG_PATTERN = r"([TUG]\s*[0O]?\d+\s*\([^)]*\))"
    TUG_PATTERN = r"([TUG]O?\s*\d+\s*\([^)]*\))"

    matches = re.findall(TUG_PATTERN, rest)

    # Normalizza e classifica
    trasformatori = []
    utenze = []
    gruppi = []

    for raw in matches:
        item = normalize_TUG(raw)

        if item.startswith("T"):
            trasformatori.append(item)
        elif item.startswith("U"):
            utenze.append(item)
        elif item.startswith("G"):
            gruppi.append(item)

        # Rimuovi dal resto
        rest = rest.replace(raw, "")

    info_text = rest.strip()

    return (
        cabina_id,
        info_text,
        " \\ ".join(trasformatori),
        " \\ ".join(utenze),
        " \\ ".join(gruppi),
    )










def normalize_single_cabin_id(raw):
    """
    DU102-262241_TR01 → DU 10-2-262241
    MB203-123456     → MB 20-3-123456
    """

    if not isinstance(raw, str):
        return None

    s = raw.upper().strip()

    # elimina suffissi tipo _TR01
    s = s.split("_")[0]

    # pattern: 2 lettere + 3 cifre + "-" + cifre
    m = re.match(r"^([A-Z]{2})(\d{3})-(\d+)$", s)
    if not m:
        return None

    prefix = m.group(1)   # DU
    num3 = m.group(2)     # 102
    tail = m.group(3)     # 262241

    part1 = num3[:2]      # 10
    part2 = num3[2]       # 2

    return f"{prefix} {part1}-{part2}-{tail}"






def normalize_cabin_id(xlsx_cabin_list):
    """
    Input: lista codici XLSX
    Output: set di codici normalizzati stile 'XX a-b-c'
    """
    normalized = set()

    for s in xlsx_cabin_list:
        #print("Codice da normalizzare: ", s)
        norm = normalize_single_cabin_id(s)
        #print("Codice normalizzato: ", norm)

        if norm:
            normalized.add(norm)
        if not norm:
            print(f"⚠️ Codice XLSX non riconosciuto: {s}")


    return normalized






def write_csv(associations, xlsx_cabin_set, csv_out: str = "associations.csv"):
    ##### OLD #####
    with open("outputs/associations_old.csv", "w", newline="", encoding="utf-8") as f:
        #writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE, escapechar='\\')

        writer.writerow(["cabina_index", "cabina_id", "cabina_conf", "info_text"])
        for a in associations:
            info_text = a["info_text"].replace("\n", " ").replace("\r", "").strip().strip('"')
            writer.writerow([a["cabina_index"], a["cabina_id"], f"{a['cabina_conf']:.1f}", info_text])



    ##### NEW #####

    '''
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        #writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE)
        writer = csv.writer(
        f,
        delimiter=";",
        quoting=csv.QUOTE_NONE,
        escapechar='\\'
        )


        writer.writerow([
            "cabina_index",
            "sigla_cabina",
            "cabina_id",
            "info_text",
            "trasformatore",
            "utenza",
            "gruppo",
            "in_xlsx",
        ])

        for a in associations:
            raw_text = a["info_text"].replace("\n", " ").replace("\r", "").strip().strip('"')

            parsed_id, info_text, trasf, ut, gr = parse_info_text(raw_text)

            writer.writerow([
                a["cabina_index"],
                a["cabina_id"],   # ← la sigla (es. MB)
                parsed_id,        # ← DU 10-2-xxxxxx
                info_text,
                trasf,
                ut,
                gr,
            ])

    '''

    new_cabin_set = normalize_cabin_id(xlsx_cabin_set)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=";",
            quoting=csv.QUOTE_NONE,
            escapechar="\\"
        )

        writer.writerow([
            "cabina_index",
            "sigla_cabina",
            "cabina_id",
            "info_text",
            "trasformatore",
            "utenza",
            "gruppo",
            "in_xlsx",
        ])

        for a in associations:
            raw_text = a["info_text"].replace("\n", " ").replace("\r", "").strip().strip('"')

            parsed_id, info_text, trasf, ut, gr = parse_info_text(raw_text)

            in_xlsx = parsed_id in new_cabin_set

            writer.writerow([
                a["cabina_index"],
                a["cabina_id"],   # ← la sigla (es. MB)
                parsed_id,        # ← DU 10-2-xxxxxx
                info_text,
                trasf,
                ut,
                gr,
                in_xlsx,  # 1 / 0
            ])