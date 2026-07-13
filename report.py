"""
Gera um relatorio pessoal a partir do session_log.csv gravado pelo main.py.

Uso: python report.py
Mostra a evolucao das suas emocoes, sinais fisiologicos/posturais
aproximados (BPM, piscadas, desvio de olhar, postura), sinais de voz
(pitch, pausas na fala) e o score de incongruencia/estresse ao longo da
sessao.

Isto e pensado para uso individual (voce revendo seu proprio dia), nao
para comparar ou ranquear pessoas diferentes, e o score de incongruencia
NAO e um indicador de mentira - ver README. Pra comparar VARIAS sessoes
ao longo do tempo, use trends.py em vez deste script.
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

PHYSIO_COLS = {
    "hr_bpm": "BPM (aprox.)",
    "blink_rate_min": "Piscadas/min",
    "gaze_offset": "Desvio de olhar",
    "shoulder_tilt_deg": "Inclinacao ombros (graus)",
}

VOICE_COLS = {
    "voice_pitch_hz": "Pitch da voz (Hz, aprox.)",
    "voice_pause_rate_min": "Pausas na fala/min",
}


def _has_data(df, col):
    return col in df.columns and df[col].notna().any()


def main():
    if not os.path.exists(LOG_PATH):
        print("Nenhum log encontrado ainda. Rode main.py e ligue o log com 'l'.")
        return

    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    if df.empty:
        print("Log vazio.")
        return

    emo_cols = list(EMOTION_LABELS_PT.keys())
    has_score = _has_data(df, "incongruence_score")
    has_physio = any(_has_data(df, c) for c in PHYSIO_COLS) or has_score
    has_voice = any(_has_data(df, c) for c in VOICE_COLS)

    n_rows = 2 + int(has_physio) + int(has_voice)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

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

    next_row = 2

    if has_physio:
        ax = axes[next_row]
        next_row += 1
        plotted = False
        for col, label in PHYSIO_COLS.items():
            if _has_data(df, col):
                ax.plot(df["timestamp"], df[col], label=label)
                plotted = True
        if plotted:
            ax.set_title("Sinais fisiologicos/postura (aprox.) ao longo da sessao")
            ax.set_ylabel("valor do sinal")
            ax.legend(loc="upper right", fontsize=8)
            ax.tick_params(axis="x", rotation=45)

        if has_score:
            ax_score = ax.twinx()
            ax_score.plot(
                df["timestamp"], df["incongruence_score"],
                color="#d62728", linestyle="--", label="Score de incongruencia",
            )
            ax_score.set_ylabel("score de incongruencia/estresse (nao e deteccao de mentira)")
            ax_score.legend(loc="upper left", fontsize=8)

    if has_voice:
        ax = axes[next_row]
        next_row += 1
        if _has_data(df, "voice_pitch_hz"):
            ax.plot(df["timestamp"], df["voice_pitch_hz"], color="#2ca02c", label="Pitch (Hz, aprox.)")
        ax.set_title("Sinal de voz (aprox.) ao longo da sessao")
        ax.set_ylabel("pitch (Hz)")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=45)

        if _has_data(df, "voice_pause_rate_min"):
            ax_pause = ax.twinx()
            ax_pause.plot(
                df["timestamp"], df["voice_pause_rate_min"],
                color="#9467bd", linestyle=":", label="Pausas na fala/min",
            )
            ax_pause.set_ylabel("pausas/min")
            ax_pause.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "session_report.png")
    plt.savefig(out_path)
    print(f"Relatorio salvo em: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
