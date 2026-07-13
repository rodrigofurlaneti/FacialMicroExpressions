"""
Gera um relatorio pessoal a partir do session_log.csv gravado pelo main.py.

Uso: python report.py
Mostra a evolucao das suas emocoes, sinais fisiologicos/posturais
aproximados (BPM, piscadas, desvio de olhar, postura) e o score de
incongruencia/estresse ao longo da sessao.

Isto e pensado para uso individual (voce revendo seu proprio dia), nao
para comparar ou ranquear pessoas diferentes, e o score de incongruencia
NAO e um indicador de mentira - ver README.
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

EXTRA_COLS = {
    "hr_bpm": "BPM (aprox.)",
    "blink_rate_min": "Piscadas/min",
    "gaze_offset": "Desvio de olhar",
    "shoulder_tilt_deg": "Inclinacao ombros (graus)",
    "incongruence_score": "Score de incongruencia/estresse",
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
    has_extra = any(c in df.columns for c in EXTRA_COLS)

    n_rows = 3 if has_extra else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))

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

    if has_extra:
        ax3 = axes[2]
        plotted = False
        for col, label in EXTRA_COLS.items():
            if col in df.columns and df[col].notna().any():
                if col == "incongruence_score":
                    continue
                ax3.plot(df["timestamp"], df[col], label=label)
                plotted = True
        if plotted:
            ax3.set_title("Sinais complementares (aprox.) ao longo da sessao")
            ax3.set_ylabel("valor do sinal")
            ax3.legend(loc="upper right", fontsize=8)
            ax3.tick_params(axis="x", rotation=45)

        if "incongruence_score" in df.columns and df["incongruence_score"].notna().any():
            ax3b = ax3.twinx()
            ax3b.plot(
                df["timestamp"], df["incongruence_score"],
                color="#d62728", linestyle="--", label="Score de incongruencia",
            )
            ax3b.set_ylabel("score de incongruencia/estresse (nao e deteccao de mentira)")
            ax3b.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "session_report.png")
    plt.savefig(out_path)
    print(f"Relatorio salvo em: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
