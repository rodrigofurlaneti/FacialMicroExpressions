"""
Compara sessoes ao longo do tempo (tendencia), juntando o session_log.csv
atual com tudo que esta arquivado em session_archive/ (criado quando voce
aperta 'c' no main.py, que agora arquiva em vez de apagar).

Uso: python trends.py

Agrupa as amostras por dia e mostra a evolucao do seu score medio de
incongruencia/estresse e da emocao mais comum, sessao a sessao. Isto
continua sendo autoconhecimento pessoal - nao serve pra comparar ou
ranquear pessoas diferentes, e o score nao e um indicador de mentira
(ver README).
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(BASE_DIR, "session_log.csv")
ARCHIVE_DIR = os.path.join(BASE_DIR, "session_archive")

EMOTION_LABELS_PT = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "surpresa",
    "neutral": "neutro",
}


def load_all_logs():
    paths = []
    if os.path.exists(LOG_PATH):
        paths.append(LOG_PATH)
    if os.path.isdir(ARCHIVE_DIR):
        paths.extend(sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.csv"))))

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[aviso] nao consegui ler {path}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def main():
    df = load_all_logs()
    if df.empty:
        print(
            "Nenhum dado encontrado. Grave pelo menos uma sessao com log ligado "
            "('l' no main.py) - sessoes anteriores arquivadas com 'c' tambem contam."
        )
        return

    df["date"] = df["timestamp"].dt.date

    agg = {"dominant_emotion": "count"}
    if "incongruence_score" in df.columns:
        agg["incongruence_score"] = "mean"

    daily = df.groupby("date").agg(agg).rename(columns={"dominant_emotion": "n_amostras"})
    daily["emocao_mais_comum"] = df.groupby("date")["dominant_emotion"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else None
    )
    daily = daily.reset_index()

    if len(daily) < 2:
        print(
            f"So encontrei dados de {len(daily)} dia(s). Precisa de pelo menos "
            f"2 dias diferentes (arquive sessoes com 'c' antes de comecar novas) "
            f"pra ver uma tendencia de verdade."
        )

    has_score = "incongruence_score" in daily.columns and daily["incongruence_score"].notna().any()
    n_rows = 2 if has_score else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    dates_str = daily["date"].astype(str)

    if has_score:
        axes[0].plot(dates_str, daily["incongruence_score"], marker="o", color="#d62728")
        axes[0].set_title("Score medio de incongruencia/estresse por dia")
        axes[0].set_ylabel("score medio (nao e deteccao de mentira)")
        axes[0].tick_params(axis="x", rotation=45)
        idx_bar = 1
    else:
        idx_bar = 0

    counts = daily["emocao_mais_comum"].map(EMOTION_LABELS_PT).value_counts()
    axes[idx_bar].bar(counts.index, counts.values, color="#3477eb")
    axes[idx_bar].set_title("Emocao mais comum - frequencia entre os dias analisados")
    axes[idx_bar].set_ylabel("numero de dias")

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "trend_report.png")
    plt.savefig(out_path)
    print(f"Relatorio de tendencia salvo em: {out_path}")

    print("\n--- Resumo por dia ---")
    for _, row in daily.iterrows():
        emo = EMOTION_LABELS_PT.get(row["emocao_mais_comum"], row["emocao_mais_comum"])
        if has_score and pd.notna(row.get("incongruence_score")):
            print(f"{row['date']}: {row['n_amostras']} amostras, emocao mais comum = {emo}, "
                  f"estresse medio = {row['incongruence_score']:.2f}")
        else:
            print(f"{row['date']}: {row['n_amostras']} amostras, emocao mais comum = {emo}, "
                  f"sem baseline calibrado nesse dia")

    plt.show()


if __name__ == "__main__":
    main()
