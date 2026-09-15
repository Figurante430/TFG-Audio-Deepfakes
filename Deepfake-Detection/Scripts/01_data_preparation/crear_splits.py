from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
DATA_DIR = ROOT / "data"

METADATA_CSV = DATA_DIR / "metadata.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
METADATA_SPLIT_CSV = DATA_DIR / "metadata_with_split.csv"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LECTURA
# ============================================================

def read_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro el metadata:\n{METADATA_CSV}"
        )

    with METADATA_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("metadata.csv está vacío.")

    required = {
        "path",
        "label",
        "category",
        "generator",
        "speaker_id",
        "utterance_id",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "Faltan columnas obligatorias en metadata.csv: "
            + ", ".join(sorted(missing))
        )

    for i, row in enumerate(rows, 1):
        row["utterance_id"] = str(row["utterance_id"]).strip()

        if not row["utterance_id"]:
            raise RuntimeError(
                f"La fila {i} no tiene utterance_id: {row.get('path')}"
            )

    return rows


# ============================================================
# SPLIT POR utterance_id
# ============================================================

def split_ids(ids, rng):
    """
    Divide una lista de utterance_id con proporción 70/15/15.
    Todos los audios que compartan utterance_id irán al mismo split.
    """
    ids = sorted(set(ids))

    rng.shuffle(ids)

    n = len(ids)

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    # El resto va a test para evitar perder elementos por redondeos.
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    assert len(test_ids) == n_test

    return train_ids, val_ids, test_ids


def build_split_map(rows):
    """
    Separamos los IDs 001-040 (los que también existen en REAL)
    del resto de IDs fake-only.

    De esta forma:
      - los 40 utterance_id con reales se reparten 28/6/6
      - los 160 restantes se reparten 112/24/24

    Con un dataset completo de 200 IDs fake y 40 IDs reales,
    esto produce exactamente:
      - 70% train
      - 15% val
      - 15% test

    tanto para REAL como para cada generador fake de 200 muestras.
    """

    all_ids = {
        row["utterance_id"]
        for row in rows
    }

    real_ids = {
        row["utterance_id"]
        for row in rows
        if str(row["label"]).strip() == "0"
    }

    fake_only_ids = all_ids - real_ids

    rng = random.Random(SEED)

    real_train, real_val, real_test = split_ids(
        list(real_ids),
        rng
    )

    fake_train, fake_val, fake_test = split_ids(
        list(fake_only_ids),
        rng
    )

    train_ids = set(real_train) | set(fake_train)
    val_ids = set(real_val) | set(fake_val)
    test_ids = set(real_test) | set(fake_test)

    split_map = {}

    for uid in train_ids:
        split_map[uid] = "train"

    for uid in val_ids:
        split_map[uid] = "val"

    for uid in test_ids:
        split_map[uid] = "test"

    # Validación: ningún utterance_id puede aparecer en dos splits.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    if set(split_map) != all_ids:
        missing = all_ids - set(split_map)
        raise RuntimeError(
            f"Hay utterance_id sin asignar: {sorted(missing)}"
        )

    print("\n================ IDs ================")
    print(f"Utterance IDs totales       : {len(all_ids)}")
    print(f"IDs con audio REAL          : {len(real_ids)}")
    print(f"IDs solo fake               : {len(fake_only_ids)}")
    print()
    print(f"IDs TRAIN                   : {len(train_ids)}")
    print(f"IDs VAL                     : {len(val_ids)}")
    print(f"IDs TEST                    : {len(test_ids)}")

    return split_map


# ============================================================
# ASIGNACIÓN
# ============================================================

def assign_splits(rows, split_map):
    result = []

    for row in rows:
        new_row = dict(row)
        new_row["split"] = split_map[row["utterance_id"]]
        result.append(new_row)

    return result


# ============================================================
# VALIDACIONES
# ============================================================

def validate_no_utterance_leakage(rows):
    by_utterance = defaultdict(set)

    for row in rows:
        by_utterance[row["utterance_id"]].add(row["split"])

    bad = {
        uid: splits
        for uid, splits in by_utterance.items()
        if len(splits) != 1
    }

    if bad:
        print("\n[ERROR] LEAKAGE DE utterance_id:")
        for uid, splits in sorted(bad.items()):
            print(f"  {uid}: {sorted(splits)}")

        raise RuntimeError(
            "Un mismo utterance_id aparece en varios splits."
        )

    print("\n[OK] No hay leakage de utterance_id.")


def validate_paths(rows):
    """
    metadata.csv guarda rutas relativas al proyecto.
    Comprueba que los WAV siguen existiendo.
    """
    missing = []

    for row in rows:
        path = Path(row["path"])

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            missing.append(str(path))

    if missing:
        print(
            f"\n[WARN] Hay {len(missing)} rutas que no existen."
        )

        for path in missing[:10]:
            print("  ", path)

        if len(missing) > 10:
            print("  ...")
    else:
        print("[OK] Todas las rutas existen.")


# ============================================================
# RESUMEN
# ============================================================

def count_by(rows, key):
    return Counter(row[key] for row in rows)


def print_counter(title, counter):
    print(f"\n{title}")

    for name, count in sorted(counter.items()):
        print(f"  {name:18s} {count}")


def print_split_summary(rows):
    print("\n============================================================")
    print("RESUMEN DE SPLITS")
    print("============================================================")

    for split in ("train", "val", "test"):
        subset = [
            row for row in rows
            if row["split"] == split
        ]

        real = sum(
            str(row["label"]).strip() == "0"
            for row in subset
        )

        fake = sum(
            str(row["label"]).strip() == "1"
            for row in subset
        )

        print()
        print(f"---------------- {split.upper()} ----------------")
        print(f"TOTAL : {len(subset)}")
        print(f"REAL  : {real}")
        print(f"FAKE  : {fake}")

        print_counter(
            "Por categoría:",
            count_by(subset, "category")
        )

        print_counter(
            "Por generador:",
            count_by(subset, "generator")
        )

        print_counter(
            "Por speaker final:",
            count_by(subset, "speaker_id")
        )


# ============================================================
# GUARDADO
# ============================================================

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def save_splits(rows):
    if not rows:
        raise RuntimeError("No hay filas que guardar.")

    fieldnames = list(rows[0].keys())

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row for row in rows
        if row["split"] == "val"
    ]

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    write_csv(
        TRAIN_CSV,
        train_rows,
        fieldnames
    )

    write_csv(
        VAL_CSV,
        val_rows,
        fieldnames
    )

    write_csv(
        TEST_CSV,
        test_rows,
        fieldnames
    )

    write_csv(
        METADATA_SPLIT_CSV,
        rows,
        fieldnames
    )

    print("\n============================================================")
    print("ARCHIVOS GENERADOS")
    print("============================================================")
    print(TRAIN_CSV)
    print(VAL_CSV)
    print(TEST_CSV)
    print(METADATA_SPLIT_CSV)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Leyendo metadata...")
    rows = read_metadata()

    print(f"Muestras encontradas: {len(rows)}")

    split_map = build_split_map(rows)

    rows = assign_splits(
        rows,
        split_map
    )

    validate_no_utterance_leakage(rows)
    validate_paths(rows)

    print_split_summary(rows)

    save_splits(rows)

    print("\nSplit terminado correctamente.")
    print(f"Seed utilizada: {SEED}")
    print("Estrategia: agrupación por utterance_id (sin leakage de frase).")


if __name__ == "__main__":
    main()
