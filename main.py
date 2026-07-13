"""
Analise de expressao facial em tempo real - uso individual e local.

Este script roda 100% na sua maquina: nenhuma imagem ou dado sai do
computador. O objetivo e autoconhecimento (ver seu proprio padrao de
humor ao longo de uma sessao), nao vigilancia ou ranking de terceiros.

Controles:
  q - sair
  l - liga/desliga o log da sessao (grava em session_log.csv)
  c - limpa o log atual
"""

import csv
import os
import time
from datetime import datetime

import cv2
from deepface import DeepFace

LOG_PATH = os.path.join(os.path.dirname(__file__), "session_log.csv")
ANALYZE_EVERY_N_FRAMES = 5  # DeepFace e pesado; nao roda a cada frame
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_LABELS_PT = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "surpresa",
    "neutral": "neutro",
}


def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "dominant_emotion"] + EMOTIONS)


def log_row(dominant_emotion, scores):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), dominant_emotion]
            + [round(scores.get(e, 0.0), 2) for e in EMOTIONS]
        )


TEXT_DARK = (20, 20, 20)
PANEL_BG = (235, 235, 235)


def draw_overlay(frame, region, dominant_emotion, scores, logging_on):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 130, 0), 2)

    label = EMOTION_LABELS_PT.get(dominant_emotion, dominant_emotion)
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    label_y = max(y - 10, label_h + 10)
    cv2.rectangle(
        frame, (x - 2, label_y - label_h - 8), (x + label_w + 6, label_y + 4),
        PANEL_BG, -1,
    )
    cv2.putText(
        frame, label, (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_DARK, 2,
    )

    panel_w = 320
    panel_h = 22 * len(EMOTIONS) + 40
    cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), PANEL_BG, -1)

    bar_x = 15
    bar_y = 35
    for emo in EMOTIONS:
        pct = scores.get(emo, 0.0)
        bar_w = int(pct * 2)  # escala: 200px = 100%
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 16), (120, 120, 120), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (200, 100, 0), -1)
        cv2.putText(
            frame, f"{EMOTION_LABELS_PT[emo]} {pct:.0f}%", (bar_x + 205, bar_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DARK, 1,
        )
        bar_y += 22

    status = "LOG: ON" if logging_on else "LOG: OFF (l p/ ligar)"
    color = (0, 0, 180) if logging_on else (90, 90, 90)
    cv2.putText(frame, status, (15, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ensure_log_header()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    frame_count = 0
    last_region = None
    last_dominant = "neutral"
    last_scores = {e: 0.0 for e in EMOTIONS}
    logging_on = False

    print("Pressione 'q' para sair, 'l' para ligar/desligar o log, 'c' para limpar o log.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, silent=True
                )
                if isinstance(result, list):
                    result = result[0]
                last_region = result["region"]
                last_scores = result["emotion"]
                last_dominant = result["dominant_emotion"]

                if logging_on and last_region["w"] > 0:
                    log_row(last_dominant, last_scores)
            except Exception as e:
                print(f"[aviso] falha na analise: {e}")

        if last_region and last_region.get("w", 0) > 0:
            draw_overlay(frame, last_region, last_dominant, last_scores, logging_on)
        else:
            cv2.putText(
                frame, "Nenhum rosto detectado", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.imshow("Analise de expressao facial (local, individual)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            logging_on = not logging_on
            print(f"Log {'ativado' if logging_on else 'desativado'}.")
        elif key == ord("c"):
            ensure_log_header()
            with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "dominant_emotion"] + EMOTIONS)
            print("Log limpo.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
