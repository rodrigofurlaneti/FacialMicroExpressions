"""
Gera um relatorio pessoal a partir do session_log.csv gravado pelo main.py.

Uso: python report.py
Mostra a evolucao das suas emocoes dominantes ao longo da sessao.
Isto e pensado para uso individual (voce revendo seu proprio dia),
nao para comparar ou ranquear pessoas diferentes.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(__file__), "session_log.csv")
EMOTION_LABELS_PT = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "surpresa",
    "neutral": "neutro",
}


def main():
    if not os.path.exists(LOG_PATH):
        print("Nenhum log encontrado ainda. Rode main.py e ligue o log com 'l'.")
        return

    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    if df.empty:
        print("Log vazio.")
        return

    emo_cols = list(EMOTION_LABELS_PT.keys())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for col in emo_cols:
        axes[0].plot(df["timestamp"], df[col], label=EMOTION_LABELS_PT[col])
    axes[0].set_title("Intensidade das emocoes ao longo da sessao")
    axes[0].set_ylabel("% de confianca")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    counts = df["dominant_emotion"].map(EMOTION_LABELS_PT).value_counts()
    axes[1].bar(counts.index, counts.values, color="#3477eb")
    axes[1].set_title("Emocao dominante - frequencia na sessao")
    axes[1].set_ylabel("numero de amostras")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "session_report.png")
    plt.savefig(out_path)
    print(f"Relatorio salvo em: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
