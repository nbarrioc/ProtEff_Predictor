# -*- coding: utf-8 -*-
"""
generate_cx_splits.py
─────────────────────────────────────────────────────────────────────────────
Genera particiones CV exhaustivas por subniveles combinatorios dentro de C1.
Garantiza el filtro biclase estricto a nivel de individuo dentro de Train.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ── Constantes de rol ─────────────────────────────────────────────────────────
TRAIN    = "train"
TEST     = "test"
EXCLUDED = "excluded"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Función de Limpieza Estricta del Train (Biclase por Individuo)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_biclase_train_iterative(
    df_train_candidate: pd.DataFrame, 
    e_col: str, 
    p_col: str, 
    y_col: str,
    min_size: int
) -> pd.DataFrame:
    """
    Filtra iterativamente el DataFrame de Train candidato hasta que CADA efector 
    y CADA proteína individual tengan al menos un positivo y un negativo DENTRO del train.
    Si el tamaño cae por debajo de min_size, aborta y devuelve un DataFrame vacío.
    """
    df_current = df_train_candidate.copy()
    
    while True:
        if len(df_current) < min_size:
            return pd.DataFrame()  # El Train se redujo demasiado, romper el fold

        # 1. Chequear efectores
        eff_counts = df_current.groupby(e_col)[y_col].agg(['sum', 'count'])
        eff_valid = eff_counts[(eff_counts['sum'] > 0) & ((eff_counts['count'] - eff_counts['sum']) > 0)].index
        
        # 2. Chequear proteínas
        prot_counts = df_current.groupby(p_col)[y_col].agg(['sum', 'count'])
        prot_valid = prot_counts[(prot_counts['sum'] > 0) & ((prot_counts['count'] - prot_counts['sum']) > 0)].index
        
        # Filtrar el dataframe actual
        df_next = df_current[df_current[e_col].isin(eff_valid) & df_current[p_col].isin(prot_valid)]
        
        # Si ya no hay cambios entre iteraciones, la matriz se ha la estabilizado
        if len(df_next) == len(df_current):
            break
            
        df_current = df_next.copy()
        
    return df_current


# ══════════════════════════════════════════════════════════════════════════════
# 2. Generación de splits combinatorios C1
# ══════════════════════════════════════════════════════════════════════════════

def generate_cx_splits(
    df: pd.DataFrame,
    effector_col:       str = "effector",
    protein_col:        str = "protein",
    label_col:          str = "label",
    sample_col:         str = "sample_name",
    min_train_ratio: float = 0.50
) -> dict:
    """
    Genera múltiples folds exhaustivos por cada subnivel de C1 evaluando de manera 
    aislada cada individuo o pareja, blindando el Train de forma biclase estricta.
    """
    e_col  = effector_col
    p_col  = protein_col
    y_col  = label_col
    sn_col = sample_col

    # Solo nos quedamos con la estructura de C1
    splits = {"C1": {}}
    
    total_samples = len(df)
    min_train_size = int(total_samples * min_train_ratio)

    def _names(dataframe):
        return dataframe[sn_col].tolist()

    # ── SUBNIVEL: C1_C2E (Test por cada Efector Individual) ───────────────────
    for eff in df[e_col].unique():
        test_mask = df[e_col] == eff
        df_test = df[test_mask]
        
        # Filtro biclase para el TEST en C2E
        if not ((df_test[y_col] == 1).any() and (df_test[y_col] == 0).any()):
            continue
            
        df_train_cand = df[~test_mask]
        df_train_clean = _filter_biclase_train_iterative(df_train_cand, e_col, p_col, y_col, min_train_size)
        
        if df_train_clean.empty:
            continue 
            
        fold_id = f"C2E_eff_{eff}"
        splits["C1"][fold_id] = {
            "train":    _names(df_train_clean),
            "test":     _names(df_test),
            "excluded": _names(df[~df[sn_col].isin(_names(df_train_clean)) & ~test_mask]),
            "meta": {
                "sublevel": "C1_C2E", "target_individual": eff,
                "n_train": len(df_train_clean), "n_test": len(df_test),
                "n_test_pos": int((df_test[y_col] == 1).sum()), "n_test_neg": int((df_test[y_col] == 0).sum())
            }
        }

    # ── SUBNIVEL: C1_C2P (Test por cada Proteína Individual) ──────────────────
    for prot in df[p_col].unique():
        test_mask = df[p_col] == prot
        df_test = df[test_mask]
        
        # Filtro biclase para el TEST en C2P
        if not ((df_test[y_col] == 1).any() and (df_test[y_col] == 0).any()):
            continue
            
        df_train_cand = df[~test_mask]
        df_train_clean = _filter_biclase_train_iterative(df_train_cand, e_col, p_col, y_col, min_train_size)
        
        if df_train_clean.empty:
            continue
            
        fold_id = f"C2P_prot_{prot}"
        splits["C1"][fold_id] = {
            "train":    _names(df_train_clean),
            "test":     _names(df_test),
            "excluded": _names(df[~df[sn_col].isin(_names(df_train_clean)) & ~test_mask]),
            "meta": {
                "sublevel": "C1_C2P", "target_individual": prot,
                "n_train": len(df_train_clean), "n_test": len(df_test),
                "n_test_pos": int((df_test[y_col] == 1).sum()), "n_test_neg": int((df_test[y_col] == 0).sum())
            }
        }

    # ── SUBNIVEL: C1_C3 (Test exhaustivo Pareja a Pareja) ─────────────────────
    for _, row_pair in df.iterrows():
        eff = row_pair[e_col]
        prot = row_pair[p_col]
        
        # Test es únicamente esta pareja (N=1, no requiere filtro biclase)
        df_test = df[(df[e_col] == eff) & (df[p_col] == prot)]
        
        # Train excluye dinámicamente al efector Y a la proteína
        df_train_cand = df[(df[e_col] != eff) & (df[p_col] != prot)]
        df_train_clean = _filter_biclase_train_iterative(df_train_cand, e_col, p_col, y_col, min_train_size)
        
        if df_train_clean.empty:
            continue
            
        fold_id = f"C3_pair_{eff}__{prot}"
        splits["C1"][fold_id] = {
            "train":    _names(df_train_clean),
            "test":     _names(df_test),
            "excluded": _names(df[~df[sn_col].isin(_names(df_train_clean)) & ~df[sn_col].isin(_names(df_test))]),
            "meta": {
                "sublevel": "C1_C3", "target_pair": f"{eff}__{prot}",
                "n_train": len(df_train_clean), "n_test": len(df_test),
                "n_test_pos": int((df_test[y_col] == 1).sum()), "n_test_neg": int((df_test[y_col] == 0).sum())
            }
        }

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 3. Reporte Adaptado
# ══════════════════════════════════════════════════════════════════════════════

def report_splits(splits: dict) -> str:
    lines = []
    def add(line=""): lines.append(line); print(line)

    SEP1 = "=" * 75
    add(SEP1)
    add("  REPORTE CV — SUBNIVELES EXHAUSTIVOS C1 CON FILTRO BICLASE ITERATIVO")
    add(SEP1)

    folds = splits["C1"]
    add(f"\nEscenario C1 ({len(folds)} folds generados exitosamente)")
    add("─" * 75)
    
    if folds:
        df_meta = pd.DataFrame([{**f['meta'], 'id': k} for k, f in folds.items()])
        for sub, sub_df in df_meta.groupby('sublevel'):
            add(f"  → Subnivel: {sub} ({len(sub_df)} folds)")
            for _, r in sub_df.head(5).iterrows():  # Muestra una pequeña vista previa (primeros 5)
                add(f"    Fold [{r['id'][:35]:<35}] train={r['n_train']:>4} test={r['n_test']:>2} (pos={r['n_test_pos']}, neg={r['n_test_neg']})")
            if len(sub_df) > 5:
                add(f"    [... y {len(sub_df)-5} folds más de este subnivel ...]")
                
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Funciones de Guardado y Carga de Splits
# ══════════════════════════════════════════════════════════════════════════════

def save_splits(splits: dict, df: pd.DataFrame, output_dir: str, sample_col: str, report_str: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_names = df[sample_col].tolist()

    folds = splits["C1"]
    if not folds:
        print("  ⚠️ Sin folds generados en C1 — no se guarda nada.")
        return

    # Matriz de roles
    roles = pd.DataFrame(index=all_names, columns=list(folds.keys()))
    roles.index.name = sample_col

    for fold_id, fold in folds.items():
        roles[fold_id] = TRAIN
        for name in fold["test"]:     roles.loc[name, fold_id] = TEST
        for name in fold["excluded"]: roles.loc[name, fold_id] = EXCLUDED

    roles.to_csv(out / "splits_C1_roles.csv")
    
    # Metadatos JSON
    meta = {fold_id: fold["meta"] for fold_id, fold in folds.items()}
    with open(out / "splits_C1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if report_str:
        with open(out / "splits_report.txt", "w", encoding="utf-8") as f:
            f.write(report_str)
    print(f"  💾 Datos guardados con éxito en {out}/")


def load_splits(output_dir: str, scenario: str = "C1") -> dict:
    """
    Carga los subniveles de C1 desde el directorio de salidas.
    """
    out = Path(output_dir)
    roles = pd.read_csv(out / "splits_C1_roles.csv", index_col=0)
    with open(out / "splits_C1_meta.json") as f:
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
# 5. Envoltura Completa (Wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_splits(
    df: pd.DataFrame,
    output_dir: str       = "splits/",
    effector_col: str       = "effector",
    protein_col: str        = "protein",
    label_col: str          = "label",
    sample_col: str         = "sample_name",
    min_train_ratio: float = 0.50
) -> dict:
    
    splits = generate_cx_splits(
        df,
        effector_col=effector_col,
        protein_col=protein_col,
        label_col=label_col,
        sample_col=sample_col,
        min_train_ratio=min_train_ratio
    )
    
    report_str = report_splits(splits)
    save_splits(splits, df, output_dir=output_dir, sample_col=sample_col, report_str=report_str)
    
    return splits