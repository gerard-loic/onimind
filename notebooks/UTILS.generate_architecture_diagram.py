"""
Génère un diagramme d'architecture du réseau CNNPlayer_v1
pour le jeu Onitama, avec justifications des choix architecturaux.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------- Configuration ----------
fig, ax = plt.subplots(1, 1, figsize=(22, 30))
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ---------- Couleurs ----------
C_INPUT    = '#4FC3F7'   # Bleu clair
C_CONV     = '#7E57C2'   # Violet
C_BN       = '#FF8A65'   # Orange
C_RELU     = '#66BB6A'   # Vert
C_RESBLOCK = '#E0E0E0'   # Gris clair (fond)
C_POLICY   = '#EF5350'   # Rouge
C_VALUE    = '#42A5F5'   # Bleu
C_DROPOUT  = '#FFCA28'   # Jaune
C_DENSE    = '#AB47BC'   # Violet foncé
C_OUTPUT   = '#26A69A'   # Teal
C_SKIP     = '#78909C'   # Gris bleu
C_JUSTIFY  = '#37474F'   # Gris foncé (texte justification)

def draw_block(ax, x, y, w, h, color, label, fontsize=9, textcolor='white', alpha=1.0, bold=False):
    """Dessine un bloc arrondi avec un label centré."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=textcolor, fontweight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.5, style='->', connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, lw=lw,
                            connectionstyle=connectionstyle,
                            mutation_scale=15)
    ax.add_patch(arrow)

def draw_annotation(ax, x, y, text, fontsize=7.5, color=C_JUSTIFY, ha='left'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            color=color, style='italic', wrap=True,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#E0E0E0', alpha=0.9))

# ---------- Titre ----------
ax.text(11, 29.3, "Architecture CNNPlayer_v1 — Onitama RL", ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1A237E')
ax.text(11, 28.8, "Inspirée d'AlphaZero : Tronc convolutionnel résiduel + double tête (Policy / Value)",
        ha='center', va='center', fontsize=10, color='#555555')

# ==================== ENTRÉE ====================
y_pos = 27.5
bw, bh = 6, 0.8
cx = 11 - bw/2

draw_block(ax, cx, y_pos, bw, bh, C_INPUT, "Input : (5, 5, 10)", fontsize=11, bold=True)

# Détail des canaux d'entrée
channel_text = (
    "10 canaux :\n"
    "  [0] Etudiants du joueur    [4] Mouvements carte 1 joueur\n"
    "  [1] Maitre du joueur         [5] Mouvements carte 2 joueur\n"
    "  [2] Etudiants adversaire  [6] Mouvements carte 1 adversaire\n"
    "  [3] Maitre adversaire       [7] Mouvements carte 2 adversaire\n"
    "                                          [8] Mouvements carte neutre\n"
    "                                          [9] Identite du joueur (+1/-1)"
)
ax.text(11, 26.5, channel_text, ha='center', va='center', fontsize=7.5,
        fontfamily='monospace', color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E1F5FE', edgecolor='#90CAF9', alpha=0.9))

draw_annotation(ax, 16.5, 26.5,
    "Pourquoi 10 canaux ?\nSeparer pieces et cartes permet\n"
    "au CNN de detecter des motifs\nspatiaux (menaces, protections)\n"
    "en croisant positions et mouvements.",
    fontsize=7)

# Flèche Input -> Conv initiale
draw_arrow(ax, 11, y_pos, 11, y_pos - 0.3)

# ==================== CONVOLUTION INITIALE ====================
y_pos = 25.0
draw_block(ax, cx, y_pos, bw, bh, C_CONV, "Conv2D(128, 3x3, same)", fontsize=10, bold=True)
draw_arrow(ax, 11, y_pos, 11, y_pos - 0.3)

y_pos -= 0.85
draw_block(ax, cx, y_pos, bw, 0.55, C_BN, "BatchNormalization", fontsize=9)
draw_arrow(ax, 11, y_pos, 11, y_pos - 0.3)

y_pos -= 0.8
draw_block(ax, cx, y_pos, bw, 0.55, C_RELU, "ReLU", fontsize=9)

draw_annotation(ax, 16.5, 24.3,
    "Convolution initiale :\n128 filtres 3x3 projettent les 10\n"
    "canaux bruts vers un espace\nde features riche. padding='same'\n"
    "preserve la grille 5x5.",
    fontsize=7)

draw_arrow(ax, 11, y_pos - 0.05, 11, y_pos - 0.45)

# ==================== BLOCS RÉSIDUELS ====================
y_res_top = 22.3
res_h = 4.2
res_w = 8
rx = 11 - res_w/2

# Fond du bloc résiduel
res_bg = FancyBboxPatch((rx, y_res_top - res_h), res_w, res_h,
                         boxstyle="round,pad=0.3",
                         facecolor=C_RESBLOCK, edgecolor='#BDBDBD',
                         linewidth=2, linestyle='--', alpha=0.6)
ax.add_patch(res_bg)

ax.text(11, y_res_top - 0.15, "Bloc Residuel  x5", ha='center', va='center',
        fontsize=12, fontweight='bold', color='#333333')

# Intérieur du bloc résiduel
inner_w = 5
inner_x = 11 - inner_w/2
y_inner = y_res_top - 0.7

draw_block(ax, inner_x, y_inner, inner_w, 0.5, C_CONV, "Conv2D(128, 3x3, same)", fontsize=8)
draw_arrow(ax, 11, y_inner, 11, y_inner - 0.3)

y_inner -= 0.65
draw_block(ax, inner_x, y_inner, inner_w, 0.45, C_BN, "BatchNorm", fontsize=8)
draw_arrow(ax, 11, y_inner, 11, y_inner - 0.3)

y_inner -= 0.6
draw_block(ax, inner_x, y_inner, inner_w, 0.45, C_RELU, "ReLU", fontsize=8)
draw_arrow(ax, 11, y_inner, 11, y_inner - 0.3)

y_inner -= 0.6
draw_block(ax, inner_x, y_inner, inner_w, 0.5, C_CONV, "Conv2D(128, 3x3, same)", fontsize=8)
draw_arrow(ax, 11, y_inner, 11, y_inner - 0.3)

y_inner -= 0.65
draw_block(ax, inner_x, y_inner, inner_w, 0.45, C_BN, "BatchNorm", fontsize=8)

# Skip connection
skip_x = inner_x + inner_w + 0.3
ax.annotate("", xy=(11 + inner_w/2 + 0.1, y_inner + 0.22),
            xytext=(11 + inner_w/2 + 0.1, y_res_top - 0.45),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2.5, connectionstyle="arc3,rad=-0.3"))
ax.text(skip_x + 0.7, (y_inner + y_res_top - 0.45)/2, "Skip\nConnection",
        ha='center', va='center', fontsize=7.5, fontweight='bold', color=C_SKIP,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_SKIP, alpha=0.9))

# Add
y_add = y_inner - 0.65
draw_block(ax, inner_x, y_add, inner_w, 0.45, '#90A4AE', "Add (x + residual)", fontsize=8, textcolor='white')
draw_arrow(ax, 11, y_add, 11, y_add - 0.3)

y_relu2 = y_add - 0.6
draw_block(ax, inner_x, y_relu2, inner_w, 0.45, C_RELU, "ReLU", fontsize=8)

# Justification des blocs résiduels
draw_annotation(ax, 16.5, 20.5,
    "Pourquoi des blocs residuels ?\n"
    "1) Evitent le vanishing gradient\n"
    "   (skip connection preserve le signal)\n"
    "2) Permettent d'empiler 5 blocs\n"
    "   sans degradation\n"
    "3) Architecture prouvee par AlphaZero\n"
    "   pour les jeux de plateau",
    fontsize=7)

draw_annotation(ax, 1.0, 20.5,
    "Pourquoi 128 filtres ?\n"
    "Compromis complexite/performance\n"
    "pour un plateau 5x5 (vs 256\n"
    "pour le Go 19x19 d'AlphaZero).\n"
    "Chaque filtre detecte un motif\n"
    "spatial different.",
    fontsize=7)

# ==================== SÉPARATION EN DEUX TÊTES ====================
y_split = 17.0
draw_arrow(ax, 11, y_relu2 - 0.05, 11, y_split + 0.5)

# Point de bifurcation
ax.plot(11, y_split + 0.3, 'o', color='#333333', markersize=8, zorder=5)
ax.text(11, y_split + 0.55, "Tronc commun  (5, 5, 128)", ha='center', va='center',
        fontsize=9, fontweight='bold', color='#555555')

# Flèches vers les deux têtes
draw_arrow(ax, 11, y_split + 0.2, 5.5, y_split - 0.3, color=C_POLICY, lw=2)
draw_arrow(ax, 11, y_split + 0.2, 16.5, y_split - 0.3, color=C_VALUE, lw=2)

# ==================== TÊTE POLICY (gauche) ====================
px = 2.2
pw = 6.5
y_p = y_split - 0.5

ax.text(px + pw/2, y_p + 0.3, "TETE POLICY (Action)", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_POLICY)

draw_block(ax, px, y_p - 0.4, pw, 0.55, C_CONV, "Conv2D(32, 1x1, same)", fontsize=8.5)
draw_arrow(ax, px + pw/2, y_p - 0.4, px + pw/2, y_p - 0.75)

draw_block(ax, px, y_p - 1.0, pw, 0.45, C_BN, "BatchNorm", fontsize=8)
draw_arrow(ax, px + pw/2, y_p - 1.0, px + pw/2, y_p - 1.3)

draw_block(ax, px, y_p - 1.5, pw, 0.45, C_RELU, "ReLU", fontsize=8)
draw_arrow(ax, px + pw/2, y_p - 1.5, px + pw/2, y_p - 1.8)

draw_block(ax, px, y_p - 2.0, pw, 0.45, C_DROPOUT, "Dropout (0.3)", fontsize=8, textcolor='#333')
draw_arrow(ax, px + pw/2, y_p - 2.0, px + pw/2, y_p - 2.3)

draw_block(ax, px, y_p - 2.55, pw, 0.55, C_CONV, "Conv2D(52, 1x1, same)", fontsize=8.5)
draw_arrow(ax, px + pw/2, y_p - 2.55, px + pw/2, y_p - 2.9)

draw_block(ax, px, y_p - 3.15, pw, 0.55, '#FFAB91', "Reshape (5x5x52 -> 1300)", fontsize=8.5, textcolor='#333')
draw_arrow(ax, px + pw/2, y_p - 3.15, px + pw/2, y_p - 3.5)

draw_block(ax, px, y_p - 3.8, pw, 0.6, C_OUTPUT,
           "Policy Logits (1300)", fontsize=10, bold=True)

# Annotations policy
draw_annotation(ax, 0.3, y_p - 5.2,
    "1300 = 5 x 5 x 52\n"
    "  5x5 : position de depart (col, row)\n"
    "  52 : index du mouvement (16 cartes)\n\n"
    "Pas de softmax dans le reseau :\n"
    "applique apres masquage des\n"
    "actions illegales (logits bruts).",
    fontsize=7)

# ==================== TÊTE VALUE (droite) ====================
vx = 13.3
vw = 6.5
y_v = y_split - 0.5

ax.text(vx + vw/2, y_v + 0.3, "TETE VALUE (Evaluation)", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_VALUE)

draw_block(ax, vx, y_v - 0.4, vw, 0.55, C_CONV, "Conv2D(4, 1x1, same)", fontsize=8.5)
draw_arrow(ax, vx + vw/2, y_v - 0.4, vx + vw/2, y_v - 0.75)

draw_block(ax, vx, y_v - 1.0, vw, 0.45, C_BN, "BatchNorm", fontsize=8)
draw_arrow(ax, vx + vw/2, y_v - 1.0, vx + vw/2, y_v - 1.3)

draw_block(ax, vx, y_v - 1.5, vw, 0.45, C_RELU, "ReLU", fontsize=8)
draw_arrow(ax, vx + vw/2, y_v - 1.5, vx + vw/2, y_v - 1.8)

draw_block(ax, vx, y_v - 2.0, vw, 0.55, '#FFAB91', "Flatten (5x5x4 -> 100)", fontsize=8.5, textcolor='#333')
draw_arrow(ax, vx + vw/2, y_v - 2.0, vx + vw/2, y_v - 2.35)

draw_block(ax, vx, y_v - 2.55, vw, 0.45, C_DROPOUT, "Dropout (0.3)", fontsize=8, textcolor='#333')
draw_arrow(ax, vx + vw/2, y_v - 2.55, vx + vw/2, y_v - 2.85)

draw_block(ax, vx, y_v - 3.1, vw, 0.55, C_DENSE, "Dense(64, ReLU)", fontsize=8.5)
draw_arrow(ax, vx + vw/2, y_v - 3.1, vx + vw/2, y_v - 3.45)

draw_block(ax, vx, y_v - 3.7, vw, 0.55, C_OUTPUT,
           "Dense(1, tanh) -> [-1, +1]", fontsize=9.5, bold=True)

# Annotations value
draw_annotation(ax, 13.5, y_p - 5.2,
    "Sortie scalaire dans [-1, +1]\n"
    "  +1 = victoire certaine\n"
    "  -1 = defaite certaine\n\n"
    "4 filtres seulement : evaluation\n"
    "globale, pas besoin d'autant de\n"
    "features que la policy.",
    fontsize=7)

# ==================== LÉGENDE DES LOSS ====================
y_loss = 7.8
loss_bg = FancyBboxPatch((1.5, y_loss - 2.2), 19, 2.5,
                          boxstyle="round,pad=0.3",
                          facecolor='#F3E5F5', edgecolor='#CE93D8',
                          linewidth=1.5, alpha=0.8)
ax.add_patch(loss_bg)

ax.text(11, y_loss + 0.05, "Entrainement", ha='center', va='center',
        fontsize=12, fontweight='bold', color='#4A148C')

ax.text(5.5, y_loss - 0.65, "Policy Loss :", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_POLICY)
ax.text(5.5, y_loss - 1.15, "CategoricalCrossentropy\n(from_logits=True)", ha='center', va='center',
        fontsize=8, color='#333333')
ax.text(5.5, y_loss - 1.75, "Poids : 1.0", ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333333')

ax.text(16.5, y_loss - 0.65, "Value Loss :", ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_VALUE)
ax.text(16.5, y_loss - 1.15, "MeanSquaredError (MSE)", ha='center', va='center',
        fontsize=8, color='#333333')
ax.text(16.5, y_loss - 1.75, "Poids : 0.5", ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333333')

ax.text(11, y_loss - 1.75,
    "Le ratio 1.0/0.5 priorise\nl'apprentissage des coups\nsur l'evaluation",
    ha='center', va='center', fontsize=7.5, color=C_JUSTIFY, style='italic',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', edgecolor='#E0E0E0'))

# ==================== JUSTIFICATIONS ARCHITECTURALES ====================
y_just = 4.5
just_bg = FancyBboxPatch((1.0, y_just - 3.5), 20, 4.0,
                          boxstyle="round,pad=0.3",
                          facecolor='#E8F5E9', edgecolor='#A5D6A7',
                          linewidth=1.5, alpha=0.8)
ax.add_patch(just_bg)

ax.text(11, y_just + 0.2, "Justifications architecturales", ha='center', va='center',
        fontsize=12, fontweight='bold', color='#1B5E20')

justifications = [
    ("Architecture AlphaZero",
     "Tronc residuel + double tete (policy/value) : architecture prouvee par DeepMind\n"
     "pour les jeux de plateau (Go, Echecs, Shogi). Le tronc partage les features\n"
     "entre les deux taches, ce qui ameliore la generalisation."),
    ("Adaptation au plateau 5x5",
     "128 filtres (vs 256 AlphaZero) et 5 blocs residuels (vs 19-39) : le plateau\n"
     "d'Onitama est bien plus petit (5x5 vs 19x19 Go), donc un reseau plus leger\n"
     "suffit et evite l'overfitting sur un espace d'etats plus restreint."),
    ("Regularisation",
     "BatchNorm (stabilite) + Dropout 0.3 (anti-overfitting) : essentiels car le\n"
     "dataset d'Onitama est plus petit que celui des jeux classiques d'AlphaZero.\n"
     "Le dropout est absent dans AlphaZero original mais necessaire ici."),
    ("Espace d'actions structure",
     "Sortie (5, 5, 52) = position x mouvement : encode naturellement la semantique\n"
     "du jeu. Le masquage des actions illegales est fait apres le reseau (logits bruts),\n"
     "ce qui permet au softmax de ne considerer que les coups legaux."),
]

y_j = y_just - 0.5
for title, desc in justifications:
    ax.text(2.0, y_j, title, ha='left', va='top',
            fontsize=8.5, fontweight='bold', color='#2E7D32')
    ax.text(2.0, y_j - 0.4, desc, ha='left', va='top',
            fontsize=7, color='#333333', fontfamily='monospace')
    y_j -= 1.0

# ==================== Légende couleurs ====================
legend_items = [
    (C_INPUT, "Input"), (C_CONV, "Convolution"), (C_BN, "BatchNorm"),
    (C_RELU, "ReLU"), (C_DROPOUT, "Dropout"), (C_DENSE, "Dense"),
    (C_OUTPUT, "Output")
]

for i, (color, label) in enumerate(legend_items):
    lx = 1.5 + i * 2.8
    ly = 0.3
    box = FancyBboxPatch((lx, ly), 0.5, 0.35, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#555555', linewidth=0.8)
    ax.add_patch(box)
    ax.text(lx + 0.7, ly + 0.18, label, ha='left', va='center', fontsize=7.5, color='#333333')

# ==================== Sauvegarde ====================
plt.tight_layout()
output_path = "/home/lgerard/python/onitama-rl/notebooks/architecture_cnnplayer_v1.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f"Diagramme sauvegardé : {output_path}")
