# -*- coding: utf-8 -*-
"""
plot_coverage_heatmap.py
─────────────────────────────────────────────────────────────────────────────
Figura 2 del esquema de selección de negativos.

Genera un heatmap 7×32 (grupos de efector × grupos de proteína) que muestra:
  · Heatmap central: cobertura de positivos y negativos por combinación de grupos.
    Cada celda muestra "n_pos | n_neg" y se colorea según su aptitud para C3
    (umbral mínimo configurable, por defecto 3 de cada clase).
  · Barplot derecho: total de positivos y negativos por grupo de efector.
    Verde si el total ≥ umbral_C2 → grupo apto para fold C2E.
  · Barplot inferior: total de positivos y negativos por grupo de proteína.
    Verde si el total ≥ umbral_C2 → grupo apto para fold C2P.

Inputs esperados
────────────────
df : pd.DataFrame con columnas:
    - 'effector'        : nombre del efector individual
    - 'protein'         : nombre de la proteína individual
    - 'effector_group'  : grupo funcional del efector (e.g. "GEF-like")
    - 'protein_group'   : grupo funcional de la proteína (e.g. "Cluster 3")
    - 'label'           : 1 (positivo) / 0 (negativo)

Autora: Lucía León Prieto / asistencia Claude (Anthropic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap


# ── Colores ──────────────────────────────────────────────────────────────────

C_GREEN       = "#1E7145"   # apto C3 (≥min_C3 pos + ≥min_C3 neg)
C_GREEN_LIGHT = "#E2EFDA"
C_ORANGE      = "#BF5000"   # ambas clases presentes pero insuficiente para C3 → train
C_ORANGE_LIGHT= "#FCE4D6"
C_RED         = "#C0392B"   # solo una clase → inferencia cualitativa
C_RED_LIGHT   = "#FADBD8"
C_GREY        = "#CCCCCC"   # sin datos
C_GREY_LIGHT  = "#F5F5F5"
C_BLUE        = "#2E75B6"   # apto C2 (barplots)
C_BLUE_LIGHT  = "#D5E8F0"
C_BAR_NONAPT  = "#F1948A"   # no apto C2 (barplots)
C_TEXT_DARK   = "#1a1a1a"
C_TEXT_MID    = "#595959"


# ── Función principal ─────────────────────────────────────────────────────────

def plot_coverage_heatmap(
    df: pd.DataFrame,
    effector_group_col: str  = "effector_group",
    protein_group_col: str   = "protein_group",
    label_col: str           = "label",
    min_C3: int              = 3,
    min_C2: int              = 3,
    title: str               = "Cobertura de positivos y negativos por combinación de grupos",
    figsize: tuple           = (24, 15),
    save_path: str           = None,
    dpi: int                 = 150,
):
    """
    Genera la Figura 2 del esquema de selección de negativos.

    Parámetros
    ──────────
    df                : DataFrame con los datos (ver cabecera del módulo).
    effector_group_col: Nombre de la columna de grupos de efector.
    protein_group_col : Nombre de la columna de grupos de proteína.
    label_col         : Nombre de la columna de etiqueta (1=positivo, 0=negativo).
    min_C3            : Umbral mínimo de positivos Y negativos por celda para
                        que esa combinación sea apta para un fold C3.
    min_C2            : Umbral mínimo de positivos Y negativos (suma por grupo)
                        para que ese grupo sea apto para un fold C2E/C2P.
    title             : Título de la figura.
    figsize           : Tamaño de la figura en pulgadas.
    save_path         : Si se especifica, guarda la figura en esa ruta.
    dpi               : Resolución de guardado.

    Retorna
    ───────
    fig, axes : Objetos matplotlib para posible modificación posterior.
    summary   : dict con conteos de celdas/grupos aptos por escenario.
    """

    # ── 1. Preparar tablas de conteo ────────────────────────────────────────────

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    # Grupos ordenados (orden de aparición en df para reproducibilidad)
    eff_groups  = list(dict.fromkeys(df[effector_group_col].dropna()))
    prot_groups = list(dict.fromkeys(df[protein_group_col].dropna()))

    n_eff  = len(eff_groups)
    n_prot = len(prot_groups)

    # Matrices n_eff × n_prot
    mat_pos = pd.DataFrame(0, index=eff_groups, columns=prot_groups, dtype=int)
    mat_neg = pd.DataFrame(0, index=eff_groups, columns=prot_groups, dtype=int)

    for _, row in pos.iterrows():
        eg = row[effector_group_col]
        pg = row[protein_group_col]
        if eg in mat_pos.index and pg in mat_pos.columns:
            mat_pos.loc[eg, pg] += 1

    for _, row in neg.iterrows():
        eg = row[effector_group_col]
        pg = row[protein_group_col]
        if eg in mat_neg.index and pg in mat_neg.columns:
            mat_neg.loc[eg, pg] += 1

    # ── 2. Clasificar cada celda ────────────────────────────────────────────────
    # 0 = sin datos
    # 1 = solo una clase (0 pos ó 0 neg)  → inferencia cualitativa
    # 2 = ambas clases pero < min_C3      → solo train
    # 3 = apto C3 (≥ min_C3 pos y neg)   → train + test C3

    color_mat = np.zeros((n_eff, n_prot), dtype=int)
    for i, eg in enumerate(eff_groups):
        for j, pg in enumerate(prot_groups):
            np_ = mat_pos.loc[eg, pg]
            nn  = mat_neg.loc[eg, pg]
            if np_ == 0 and nn == 0:
                color_mat[i, j] = 0   # sin datos
            elif np_ == 0 or nn == 0:
                color_mat[i, j] = 1   # solo una clase
            elif np_ >= min_C3 and nn >= min_C3:
                color_mat[i, j] = 3   # apto C3
            else:
                color_mat[i, j] = 2   # ambas clases, insuficiente C3

    # ── 3. Totales para barplots ────────────────────────────────────────────────
    # Solo se cuentan parejas de celdas con ambas clases (estados 2 y 3)

    def total_with_both_classes(group_pos, group_neg):
        """Suma pos y neg solo de celdas que tienen al menos 1 de cada clase."""
        total_p, total_n = 0, 0
        for p, n in zip(group_pos, group_neg):
            if p > 0 and n > 0:
                total_p += p
                total_n += n
        return total_p, total_n

    # Por grupo de efector
    eff_pos_total, eff_neg_total, eff_apt_C2E = [], [], []
    for eg in eff_groups:
        row_pos = [mat_pos.loc[eg, pg] for pg in prot_groups]
        row_neg = [mat_neg.loc[eg, pg] for pg in prot_groups]
        tp, tn = total_with_both_classes(row_pos, row_neg)
        eff_pos_total.append(tp)
        eff_neg_total.append(tn)
        eff_apt_C2E.append(tp >= min_C2 and tn >= min_C2)

    # Por grupo de proteína
    prot_pos_total, prot_neg_total, prot_apt_C2P = [], [], []
    for pg in prot_groups:
        col_pos = [mat_pos.loc[eg, pg] for eg in eff_groups]
        col_neg = [mat_neg.loc[eg, pg] for eg in eff_groups]
        tp, tn = total_with_both_classes(col_pos, col_neg)
        prot_pos_total.append(tp)
        prot_neg_total.append(tn)
        prot_apt_C2P.append(tp >= min_C2 and tn >= min_C2)

    # ── 4. Layout con GridSpec ──────────────────────────────────────────────────
    # Proporciones: heatmap ocupa la mayor parte; barplots son más estrechos

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[n_prot, max(3, n_prot // 5)],   # heatmap ancho, barplot derecho estrecho
        height_ratios=[n_eff, max(2, n_eff // 3)],    # heatmap alto, barplot inferior estrecho
        hspace=0.40,
        wspace=0.10,
    )

    ax_heat   = fig.add_subplot(gs[0, 0])   # heatmap central
    ax_right  = fig.add_subplot(gs[0, 1])   # barplot derecho  (C2E)
    ax_bottom = fig.add_subplot(gs[1, 0])   # barplot inferior (C2P)
    ax_legend = fig.add_subplot(gs[1, 1])   # leyenda
    ax_legend.axis("off")

    # ── 5. Dibujar heatmap ──────────────────────────────────────────────────────

    cmap = ListedColormap([C_GREY_LIGHT, C_RED_LIGHT, C_ORANGE_LIGHT, C_GREEN_LIGHT])

    ax_heat.imshow(
        color_mat,
        cmap=cmap,
        vmin=0, vmax=3,
        aspect="auto",
        interpolation="nearest",
    )

    # Bordes de celdas
    for i in range(n_eff + 1):
        ax_heat.axhline(i - 0.5, color="white", linewidth=1.2)
    for j in range(n_prot + 1):
        ax_heat.axvline(j - 0.5, color="white", linewidth=1.2)

    # Texto en cada celda: "n_pos | n_neg"
    for i, eg in enumerate(eff_groups):
        for j, pg in enumerate(prot_groups):
            np_ = mat_pos.loc[eg, pg]
            nn  = mat_neg.loc[eg, pg]
            if np_ > 0 or nn > 0:
                state = color_mat[i, j]
                txt_color = (C_GREEN   if state == 3 else
                             C_ORANGE  if state == 2 else
                             C_RED     if state == 1 else C_TEXT_MID)
                ax_heat.text(
                    j, i,
                    f"{np_} | {nn}",
                    ha="center", va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=txt_color,
                )

    # Ejes del heatmap
    ax_heat.set_xticks(range(n_prot))
    ax_heat.set_xticklabels(prot_groups, rotation=45, ha="right", fontsize=15)
    ax_heat.set_yticks(range(n_eff))
    ax_heat.set_yticklabels(eff_groups, fontsize=16)
    ax_heat.set_xlabel("Grupos de proteína huésped", fontsize=18, labelpad=8)
    ax_heat.set_ylabel("Grupos de efector", fontsize=18, labelpad=8)
    ax_heat.xaxis.set_label_position("bottom")
    ax_heat.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # ── 6. Barplot derecho: totales por grupo de efector (C2E) ─────────────────

    bar_colors_right = [C_BLUE if apt else C_BAR_NONAPT for apt in eff_apt_C2E]
    y_pos = np.arange(n_eff)

    bars_pos = ax_right.barh(
        y_pos, eff_pos_total,
        color=bar_colors_right, alpha=0.85,
        height=0.6, label="Positivos", align="center",
    )
    bars_neg = ax_right.barh(
        y_pos, eff_neg_total,
        color=bar_colors_right, alpha=0.45,
        height=0.6, label="Negativos", align="center",
        left=eff_pos_total,
    )

    # Anotaciones de conteo
    for idx, (p, n, apt) in enumerate(zip(eff_pos_total, eff_neg_total, eff_apt_C2E)):
        total = p + n
        label_color = C_BLUE if apt else C_ORANGE
        ax_right.text(
            total + 0.3, idx,
            f"{p}+{n}",
            va="center", ha="left", fontsize=18,
            color=label_color, fontweight="bold",
        )

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([])
    ax_right.set_ylim(-0.5, n_eff - 0.5)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Total parejas\n(pos + neg)", fontsize=14, labelpad=6)
    ax_right.set_title("C2E\napt.", fontsize=15, color=C_TEXT_MID, pad=4)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.tick_params(left=False)

    # Línea de umbral C2
    ax_right.axvline(
        min_C2 * 2, color=C_BLUE, linewidth=1,
        linestyle="--", alpha=0.6,
        label=f"umbral C2 ({min_C2}+{min_C2})",
    )

    # ── 7. Barplot inferior: totales por grupo de proteína (C2P) ───────────────

    bar_colors_bottom = [C_BLUE if apt else C_BAR_NONAPT for apt in prot_apt_C2P]
    x_pos = np.arange(n_prot)

    ax_bottom.bar(
        x_pos, prot_pos_total,
        color=bar_colors_bottom, alpha=0.85,
        width=0.5, label="Positivos", align="center",
    )
    ax_bottom.bar(
        x_pos, prot_neg_total,
        color=bar_colors_bottom, alpha=0.45,
        width=0.5, label="Negativos", align="center",
        bottom=prot_pos_total,
    )

    # Anotaciones rotadas
    for idx, (p, n, apt) in enumerate(zip(prot_pos_total, prot_neg_total, prot_apt_C2P)):
        total = p + n
        if total > 0:
            label_color = C_BLUE if apt else C_ORANGE
            ax_bottom.text(
                idx, total + 0.2,
                f"{p}+{n}",
                ha="center", va="bottom",
                fontsize=16, rotation=90,
                color=label_color, fontweight="bold",
            )

    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels([])
    ax_bottom.set_xlim(-0.5, n_prot - 0.5)
    ax_bottom.set_ylabel("Total parejas\n(pos + neg)", fontsize=14, labelpad=6)
    ax_bottom.set_title("C2P aptitud", fontsize=15, color=C_TEXT_MID, pad=30)
    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.axhline(
        min_C2 * 2, color=C_BLUE, linewidth=1,
        linestyle="--", alpha=0.6,
    )

    # ── 8. Leyenda ──────────────────────────────────────────────────────────────

    legend_patches = [
        mpatches.Patch(facecolor=C_GREEN_LIGHT, edgecolor=C_GREEN, linewidth=1.5,
                       label=f"Apto C3  (≥{min_C3} pos + ≥{min_C3} neg)"),
        mpatches.Patch(facecolor=C_ORANGE_LIGHT, edgecolor=C_ORANGE, linewidth=1.5,
                       label=f"Ambas clases, insuficiente C3 → solo train"),
        mpatches.Patch(facecolor=C_RED_LIGHT, edgecolor=C_RED, linewidth=1.5,
                       label="Solo una clase → inferencia cualitativa"),
        mpatches.Patch(facecolor=C_GREY_LIGHT, edgecolor=C_GREY, linewidth=1.5,
                       label="Sin datos"),
        mpatches.Patch(facecolor=C_BLUE_LIGHT, edgecolor=C_BLUE, linewidth=1.5,
                       label=f"Grupo apto C2E/C2P (≥{min_C2}+{min_C2} total, ambas clases)"),
        mpatches.Patch(facecolor=C_RED_LIGHT, edgecolor=C_BAR_NONAPT, linewidth=1.5,
                       label="Grupo no apto para C2"),
    ]

    ax_legend.legend(
        handles=legend_patches,
        loc="center",
        frameon=True,
        framealpha=0.9,
        fontsize=17,
        title="Leyenda",
        title_fontsize=18,
    )

    # Nota sobre el texto de celdas
    ax_legend.text(
        0.5, 0.08,
        'Texto en celda: "n_pos | n_neg"\n(positivos | negativos disponibles)',
        ha="center", va="center",
        fontsize=16, color=C_TEXT_MID,
        transform=ax_legend.transAxes,
    )

    # ── 9. Título y resumen ─────────────────────────────────────────────────────

    n_C3_apt      = int((color_mat == 3).sum())
    n_C3_train    = int((color_mat == 2).sum())
    n_one_class   = int((color_mat == 1).sum())
    n_C3_empty    = int((color_mat == 0).sum())
    n_C2E_apt     = sum(eff_apt_C2E)
    n_C2P_apt     = sum(prot_apt_C2P)

    subtitle = (
        f"Aptas C3: {n_C3_apt}  ·  "
        f"Solo train (ambas clases): {n_C3_train}  ·  "
        f"Inferencia cualitativa (1 clase): {n_one_class}  ·  "
        f"Sin datos: {n_C3_empty}  ·  "
        f"Grupos efector aptos C2E: {n_C2E_apt}/{n_eff}  ·  "
        f"Grupos proteína aptos C2P: {n_C2P_apt}/{n_prot}"
    )

    fig.suptitle(title, fontsize=22, fontweight="bold", color=C_TEXT_DARK, y=1.01)
    fig.text(0.5, 0.995, subtitle, ha="center", fontsize=14, color=C_TEXT_MID,
             transform=fig.transFigure)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figura guardada en: {save_path}")

    # ── 10. Resumen por consola ─────────────────────────────────────────────────

    summary = {
        "n_cells_C3_apt":          n_C3_apt,
        "n_cells_train_only":      n_C3_train,
        "n_cells_one_class":       n_one_class,
        "n_cells_empty":           n_C3_empty,
        "n_effector_groups_C2E":   n_C2E_apt,
        "n_protein_groups_C2P":    n_C2P_apt,
        "effector_apt_C2E":        [eg for eg, apt in zip(eff_groups, eff_apt_C2E) if apt],
        "effector_nonapt_C2E":     [eg for eg, apt in zip(eff_groups, eff_apt_C2E) if not apt],
        "protein_apt_C2P":         [pg for pg, apt in zip(prot_groups, prot_apt_C2P) if apt],
        "protein_nonapt_C2P":      [pg for pg, apt in zip(prot_groups, prot_apt_C2P) if not apt],
    }

    print("\n── Resumen de aptitud por escenario ──────────────────────────────")
    print(f"  C3  : {n_C3_apt} combinaciones aptas (verde)")
    print(f"  Train only: {n_C3_train} combinaciones (naranja, ambas clases, <{min_C3})")
    print(f"  Inf. cualitativa: {n_one_class} combinaciones (rojo, solo 1 clase)")
    print(f"  Sin datos: {n_C3_empty} combinaciones (gris)")
    print(f"  C2E : {n_C2E_apt}/{n_eff} grupos de efector aptos")
    print(f"  C2P : {n_C2P_apt}/{n_prot} grupos de proteína aptos")
    if summary["effector_nonapt_C2E"]:
        print(f"  ⚠️  Grupos efector NO aptos C2E: {summary['effector_nonapt_C2E']}")
    if summary["protein_nonapt_C2P"]:
        print(f"  ⚠️  Grupos proteína NO aptos C2P: {summary['protein_nonapt_C2P']}")
    print("──────────────────────────────────────────────────────────────────\n")

    return fig, (ax_heat, ax_right, ax_bottom, ax_legend), summary


# ── Ejemplo de uso con datos sintéticos ──────────────────────────────────────

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    eff_groups_ex  = [f"EG{i}" for i in range(1, 8)]   # 7 grupos de efector
    prot_groups_ex = [f"PG{i}" for i in range(1, 12)]  # 11 grupos de proteína (más manejable)

    rows = []
    for eg in eff_groups_ex:
        for pg in prot_groups_ex:
            # Algunas combinaciones sin datos, algunas con pocos, algunas con suficientes
            n_pos = rng.choice([0, 0, 1, 2, 3, 4, 5, 6], p=[0.25, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05])
            n_neg = rng.choice([0, 0, 1, 2, 3, 4, 5, 6], p=[0.20, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.10])
            for _ in range(n_pos):
                rows.append({"effector": f"eff_{eg}", "protein": f"prot_{pg}",
                             "effector_group": eg, "protein_group": pg, "label": 1})
            for _ in range(n_neg):
                rows.append({"effector": f"eff_{eg}", "protein": f"prot_{pg}",
                             "effector_group": eg, "protein_group": pg, "label": 0})

    df_example = pd.DataFrame(rows)
    print(f"Dataset sintético: {len(df_example)} parejas "
          f"({(df_example.label==1).sum()} pos, {(df_example.label==0).sum()} neg)")

    fig, axes, summary = plot_coverage_heatmap(
        df_example,
        min_C3=3,
        min_C2=3,
        title="Cobertura de positivos y negativos por combinación de grupos [EJEMPLO]",
        save_path="/figura_ejemplo_coverage_heatmap.png",
        dpi=150,
    )
    plt.show()
