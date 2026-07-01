# -*- coding: utf-8 -*-
"""
generate_cx_splits.py
─────────────────────────────────────────────────────────────────────────────
Genera, reporta y guarda las particiones CV para los tres escenarios Cx:

  · C3  : leave-one-(effector_group × protein_group)-out
          Solo combinaciones con ≥ min_C3 positivos Y ≥ min_C3 negativos.
          Excluidos de cada fold: pares que comparten el grupo de efector
          O el grupo de proteína con la combinación de test (pero no ambos).

  · C2E : leave-one-effector_group-out
          Solo grupos de efector con ≥ min_C2 pos Y ≥ min_C2 neg en total
          (contando únicamente celdas que tengan ambas clases).
          Test set = todos los pares del grupo de efector dejado fuera.

  · C2P : leave-one-protein_group-out  (simétrico a C2E)

Formato de salida
─────────────────
Por escenario se generan dos archivos en output_dir/:

  splits_C3_roles.csv   ← matriz de roles (filas=muestras, columnas=folds)
  splits_C3_meta.json   ← metadatos de cada fold (grupos, n_pos, n_neg)
  splits_C2E_roles.csv
  splits_C2E_meta.json
  splits_C2P_roles.csv
  splits_C2P_meta.json
  splits_report.txt     ← resumen completo por pantalla y en archivo

Valores de roles:
  "test"     → esta muestra es test en este fold
  "train"    → esta muestra va a train en este fold
  "excluded" → esta muestra se excluye de este fold (solo ocurre en C3)

Carga en código de entrenamiento
─────────────────────────────────
  from generate_cx_splits import load_splits

  folds = load_splits("splits/", scenario="C2E")
  for fold_id, fold in folds.items():
      train_names = fold["train"]   # lista de sample_name
      test_names  = fold["test"]    # lista de sample_name

Inputs esperados
────────────────
df : pd.DataFrame con columnas:
    - sample_name    : identificador único de la pareja (e.g. "ProtA_EffB_1")
    - effector       : nombre del efector individual
    - protein        : nombre de la proteína individual
    - effector_group : grupo funcional del efector
    - protein_group  : grupo funcional de la proteína
    - label          : 1 (positivo) / 0 (negativo)

Autora: Lucía León Prieto / asistencia Claude (Anthropic)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constantes de rol ─────────────────────────────────────────────────────────
TRAIN    = "train"
TEST     = "test"
EXCLUDED = "excluded"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Generación de splits
# ══════════════════════════════════════════════════════════════════════════════

def generate_cx_splits(
    df: pd.DataFrame,
    effector_group_col: str = "effector_group",
    protein_group_col:  str = "protein_group",
    label_col:          str = "label",
    sample_col:         str = "sample_name",
    min_C3: int = 3,
    min_C2: int = 3,
) -> dict:
    """
    Genera todos los splits válidos para los escenarios C3, C2E y C2P.

    Parámetros
    ──────────
    df               : DataFrame con los datos de entrenamiento
                       (solo pares con ambas clases representadas,
                        i.e. celdas verdes + naranjas del heatmap P2).
    min_C3           : Mínimo de pos Y neg en una combinación para
                       que sea elegible como fold de test en C3.
    min_C2           : Mínimo de pos Y neg en el agregado de un grupo
                       para que sea elegible como fold de test en C2E/C2P.

    Retorna
    ───────
    splits : dict con claves "C3", "C2E", "C2P".
             Cada valor es un dict {fold_id (str): {"train": [...], "test": [...],
             "excluded": [...], "meta": {...}}}.
    """

    eg_col = effector_group_col
    pg_col = protein_group_col
    y_col  = label_col
    sn_col = sample_col

    splits = {"C3": {}, "C2E": {}, "C2P": {}}

    all_names = df[sn_col].tolist()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _counts(mask):
        sub = df[mask]
        return int((sub[y_col] == 1).sum()), int((sub[y_col] == 0).sum())

    def _valid(n_pos, n_neg, threshold):
        return n_pos >= threshold and n_neg >= threshold

    def _names(mask):
        return df[mask][sn_col].tolist()

    # ── C3 : leave-one-(eg × pg)-out ─────────────────────────────────────────

    combos = df.groupby([eg_col, pg_col])
    for (eg, pg), _ in combos:
        test_mask = (df[eg_col] == eg) & (df[pg_col] == pg)
        n_pos, n_neg = _counts(test_mask)

        if not _valid(n_pos, n_neg, min_C3):
            continue  # combinación no apta para test C3

        # Excluidos: comparten SOLO efector O SOLO proteína con el test
        excl_mask = (
            ((df[eg_col] == eg) & (df[pg_col] != pg)) |
            ((df[eg_col] != eg) & (df[pg_col] == pg))
        )
        train_mask = ~test_mask & ~excl_mask

        fold_id = f"{eg}__{pg}"
        splits["C3"][fold_id] = {
            "train":    _names(train_mask),
            "test":     _names(test_mask),
            "excluded": _names(excl_mask),
            "meta": {
                "effector_group": eg,
                "protein_group":  pg,
                "n_train":        int(train_mask.sum()),
                "n_test":         int(test_mask.sum()),
                "n_test_pos":     n_pos,
                "n_test_neg":     n_neg,
                "n_excluded":     int(excl_mask.sum()),
            }
        }

    # ── C2E : leave-one-effector_group-out ───────────────────────────────────

    for eg in df[eg_col].unique():
        test_mask = df[eg_col] == eg

        # Solo contar celdas con ambas clases para el umbral C2
        sub = df[test_mask]
        combos_eg = sub.groupby(pg_col)
        n_pos_valid = sum(
            int((g[y_col] == 1).sum())
            for _, g in combos_eg
            if (g[y_col] == 1).any() and (g[y_col] == 0).any()
        )
        n_neg_valid = sum(
            int((g[y_col] == 0).sum())
            for _, g in combos_eg
            if (g[y_col] == 1).any() and (g[y_col] == 0).any()
        )

        if not _valid(n_pos_valid, n_neg_valid, min_C2):
            continue  # grupo no apto para test C2E

        train_mask = ~test_mask
        n_pos_test, n_neg_test = _counts(test_mask)

        splits["C2E"][eg] = {
            "train":    _names(train_mask),
            "test":     _names(test_mask),
            "excluded": [],
            "meta": {
                "effector_group": eg,
                "n_train":        int(train_mask.sum()),
                "n_test":         int(test_mask.sum()),
                "n_test_pos":     n_pos_test,
                "n_test_neg":     n_neg_test,
            }
        }

    # ── C2P : leave-one-protein_group-out ────────────────────────────────────

    for pg in df[pg_col].unique():
        test_mask = df[pg_col] == pg

        sub = df[test_mask]
        combos_pg = sub.groupby(eg_col)
        n_pos_valid = sum(
            int((g[y_col] == 1).sum())
            for _, g in combos_pg
            if (g[y_col] == 1).any() and (g[y_col] == 0).any()
        )
        n_neg_valid = sum(
            int((g[y_col] == 0).sum())
            for _, g in combos_pg
            if (g[y_col] == 1).any() and (g[y_col] == 0).any()
        )

        if not _valid(n_pos_valid, n_neg_valid, min_C2):
            continue

        train_mask = ~test_mask
        n_pos_test, n_neg_test = _counts(test_mask)

        splits["C2P"][pg] = {
            "train":    _names(train_mask),
            "test":     _names(test_mask),
            "excluded": [],
            "meta": {
                "protein_group": pg,
                "n_train":       int(train_mask.sum()),
                "n_test":        int(test_mask.sum()),
                "n_test_pos":    n_pos_test,
                "n_test_neg":    n_neg_test,
            }
        }

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 2. Reporte
# ══════════════════════════════════════════════════════════════════════════════

def report_splits(splits: dict, min_C3: int = 3, min_C2: int = 3) -> str:
    """
    Genera el reporte completo de splits en texto.
    Retorna el string del reporte (también imprime por pantalla).
    """
    lines = []

    def add(line=""):
        lines.append(line)
        print(line)

    SEP1 = "=" * 65
    SEP2 = "─" * 65

    add(SEP1)
    add("  REPORTE DE PARTICIONES CV — ESCENARIOS Cx")
    add(f"  Umbral C3: ≥{min_C3} pos + ≥{min_C3} neg por combinación")
    add(f"  Umbral C2: ≥{min_C2} pos + ≥{min_C2} neg por grupo (celdas biclase)")
    add(SEP1)

    for scenario in ["C3", "C2E", "C2P"]:
        folds = splits[scenario]
        add(f"\n{'─'*65}")
        add(f"  Escenario {scenario}  ({len(folds)} folds válidos)")
        add(f"{'─'*65}")

        if not folds:
            add("  ⚠️  Sin folds válidos para este escenario.")
            continue

        for fold_id, fold in folds.items():
            m = fold["meta"]
            pos = m["n_test_pos"]
            neg = m["n_test_neg"]
            excl = m.get("n_excluded", 0)
            flag = "  ✅" if pos >= min_C3 and neg >= min_C3 else "  ⚠️"

            if scenario == "C3":
                header = (f"  Fold [{fold_id:<30}]"
                          f"  train={m['n_train']:>3}"
                          f"  test={m['n_test']:>3} (pos={pos}, neg={neg})"
                          f"  excl={excl:>3}{flag}")
            elif scenario == "C2E":
                header = (f"  Fold [efector_grp={fold_id:<20}]"
                          f"  train={m['n_train']:>3}"
                          f"  test={m['n_test']:>3} (pos={pos}, neg={neg}){flag}")
            else:
                header = (f"  Fold [protein_grp={fold_id:<20}]"
                          f"  train={m['n_train']:>3}"
                          f"  test={m['n_test']:>3} (pos={pos}, neg={neg}){flag}")

            add(header)

        # Resumen del escenario
        n_valid = sum(
            1 for f in folds.values()
            if f["meta"]["n_test_pos"] > 0 and f["meta"]["n_test_neg"] > 0
        )
        add(f"\n  → Folds con ambas clases en test: {n_valid}/{len(folds)}")

    add(f"\n{SEP1}")
    add("  RESUMEN GLOBAL")
    add(SEP2)
    for scenario in ["C3", "C2E", "C2P"]:
        n = len(splits[scenario])
        status = "✅ viable" if n >= 3 else ("⚠️  pocos folds" if n > 0 else "❌ inviable")
        add(f"  {scenario:<5}: {n:>3} folds  →  {status}")
    add(SEP1)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Guardado
# ══════════════════════════════════════════════════════════════════════════════

def save_splits(
    splits: dict,
    df: pd.DataFrame,
    output_dir: str = "splits/",
    sample_col: str = "sample_name",
    report_str: str = "",
):
    """
    Guarda los splits en output_dir/:

      splits_C3_roles.csv   ← matriz roles (muestra × fold)
      splits_C3_meta.json   ← metadatos de cada fold
      splits_C2E_roles.csv / splits_C2E_meta.json
      splits_C2P_roles.csv / splits_C2P_meta.json
      splits_report.txt     ← reporte completo

    La matriz de roles tiene:
      filas   = todas las muestras del df
      columnas = un fold por columna
      valores  = "train" | "test" | "excluded"
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_names = df[sample_col].tolist()

    for scenario, folds in splits.items():

        if not folds:
            print(f"  [{scenario}] Sin folds — no se guarda.")
            continue

        # ── Matriz de roles ──────────────────────────────────────────────────
        roles = pd.DataFrame(index=all_names, columns=list(folds.keys()))
        roles.index.name = sample_col

        for fold_id, fold in folds.items():
            # Inicializar todo como train
            roles[fold_id] = TRAIN
            # Marcar test
            for name in fold["test"]:
                roles.loc[name, fold_id] = TEST
            # Marcar excluidos (solo C3)
            for name in fold["excluded"]:
                roles.loc[name, fold_id] = EXCLUDED

        roles_path = out / f"splits_{scenario}_roles.csv"
        roles.to_csv(roles_path)
        print(f"  Guardado: {roles_path}  "
              f"({len(all_names)} muestras × {len(folds)} folds)")

        # ── Metadatos ────────────────────────────────────────────────────────
        meta = {fold_id: fold["meta"] for fold_id, fold in folds.items()}
        meta_path = out / f"splits_{scenario}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Guardado: {meta_path}")

    # ── Reporte ──────────────────────────────────────────────────────────────
    if report_str:
        report_path = out / "splits_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"  Guardado: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Carga en código de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(output_dir: str, scenario: str) -> dict:
    """
    Carga los splits de un escenario desde output_dir/.

    Parámetros
    ──────────
    output_dir : directorio donde se guardaron los splits.
    scenario   : "C3", "C2E" o "C2P".

    Retorna
    ───────
    folds : dict {fold_id: {"train": [sample_names],
                             "test":  [sample_names],
                             "meta":  {...}}}

    Ejemplo de uso en entrenamiento
    ────────────────────────────────
    folds = load_splits("splits/", "C2E")
    for fold_id, fold in folds.items():
        train_names = fold["train"]
        test_names  = fold["test"]
        # filtrar df por nombres → obtener X_train, y_train, X_test, y_test
    """
    out = Path(output_dir)

    roles_path = out / f"splits_{scenario}_roles.csv"
    meta_path  = out / f"splits_{scenario}_meta.json"

    if not roles_path.exists():
        raise FileNotFoundError(f"No se encontró {roles_path}")

    roles = pd.read_csv(roles_path, index_col=0)

    with open(meta_path) as f:
        meta = json.load(f)

    folds = {}
    for fold_id in roles.columns:
        col = roles[fold_id]
        folds[fold_id] = {
            "train": col[col == TRAIN].index.tolist(),
            "test":  col[col == TEST].index.tolist(),
            "meta":  meta.get(fold_id, {}),
        }

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# 5. Función wrapper: genera + reporta + guarda
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_splits(
    df: pd.DataFrame,
    output_dir: str       = "splits/",
    effector_group_col: str = "effector_group",
    protein_group_col: str  = "protein_group",
    label_col: str          = "label",
    sample_col: str         = "sample_name",
    min_C3: int = 3,
    min_C2: int = 3,
) -> dict:
    """
    Wrapper que ejecuta el pipeline completo:
      1. Genera todos los splits Cx
      2. Imprime y guarda el reporte
      3. Guarda los CSVs y JSONs

    Retorna el dict de splits para uso inmediato.
    """
    print("\n🔧 Generando splits Cx...")
    splits = generate_cx_splits(
        df,
        effector_group_col=effector_group_col,
        protein_group_col=protein_group_col,
        label_col=label_col,
        sample_col=sample_col,
        min_C3=min_C3,
        min_C2=min_C2,
    )

    print("\n📋 Reporte de particiones:\n")
    report_str = report_splits(splits, min_C3=min_C3, min_C2=min_C2)

    print("\n💾 Guardando splits...")
    save_splits(splits, df, output_dir=output_dir,
                sample_col=sample_col, report_str=report_str)

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# Ejemplo de uso con datos sintéticos
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    eff_groups_ex  = [f"EG{i}" for i in range(1, 8)]
    prot_groups_ex = [f"PG{i}" for i in range(1, 12)]

    rows = []
    for eg in eff_groups_ex:
        for pg in prot_groups_ex:
            n_pos = rng.choice([0, 1, 2, 3, 4, 5],
                               p=[0.20, 0.10, 0.15, 0.20, 0.20, 0.15])
            n_neg = rng.choice([0, 1, 2, 3, 4, 5],
                               p=[0.20, 0.10, 0.15, 0.20, 0.20, 0.15])
            # Solo incluimos celdas con ambas clases (verdes + naranjas)
            if n_pos == 0 or n_neg == 0:
                continue
            for k in range(n_pos):
                rows.append({
                    "sample_name":    f"{eg}_{pg}_pos_{k}",
                    "effector":       f"eff_{eg}",
                    "protein":        f"prot_{pg}",
                    "effector_group": eg,
                    "protein_group":  pg,
                    "label":          1,
                })
            for k in range(n_neg):
                rows.append({
                    "sample_name":    f"{eg}_{pg}_neg_{k}",
                    "effector":       f"eff_{eg}",
                    "protein":        f"prot_{pg}",
                    "effector_group": eg,
                    "protein_group":  pg,
                    "label":          0,
                })

    df_example = pd.DataFrame(rows)
    print(f"Dataset sintético: {len(df_example)} parejas  "
          f"({(df_example.label==1).sum()} pos, "
          f"{(df_example.label==0).sum()} neg)")

    splits = build_and_save_splits(
        df_example,
        output_dir="/mnt/user-data/outputs/splits/",
        min_C3=3,
        min_C2=3,
    )

    # ── Ejemplo de carga en código de entrenamiento ───────────────────────────
    print("\n── Ejemplo de carga ──────────────────────────────────────────────")
    folds_C2E = load_splits("/mnt/user-data/outputs/splits/", "C2E")
    for fold_id, fold in list(folds_C2E.items())[:2]:
        print(f"\n  Fold C2E [{fold_id}]")
        print(f"    train : {len(fold['train'])} muestras")
        print(f"    test  : {len(fold['test'])} muestras")
        print(f"    meta  : {fold['meta']}")
