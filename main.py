"""
Analise de expressao facial em tempo real - uso individual e local.

Este script roda 100% na sua maquina: nenhuma imagem ou dado sai do
computador. O objetivo e autoconhecimento (ver seu proprio padrao de
humor/estresse ao longo de uma sessao), nao vigilancia, ranking de
terceiros ou deteccao de mentira. Veja o README para o porque disso.

Sinais capturados:
  - Emocao dominante (DeepFace): macro-expressoes.
  - Piscadas e desvio de olhar (MediaPipe Face Mesh).
  - Frequencia cardiaca aproximada via rPPG (MediaPipe + testa).
  - Postura: inclinacao dos ombros e proxy de inclinar o tronco
    (MediaPipe Pose).
  - Um "score de incongruencia/estresse", que so existe DEPOIS de
    calibrar seu proprio baseline neutro (tecla 'b'). Ele mede o quanto
    voce se afastou do SEU proprio normal - nao indica mentira.

Controles:
  q - sair
  l - liga/desliga o log da sessao (grava em session_log.csv)
  c - limpa o log atual
  b - (re)inicia a calibracao do baseline (fique parado, expressao neutra)
"""

import collections
import csv
import os
import time
from datetime import datetime

import cv2
from deepface import DeepFace

try:
    from signals import BaselineCalibrator, FaceMeshTracker, PoseTracker, RPPGEstimator
    SIGNALS_AVAILABLE = True
except ImportError as e:
    print(f"[aviso] sinais complementares desativados ({e}). "
          f"Rode: pip install -r requirements.txt")
    SIGNALS_AVAILABLE = False

LOG_PATH = os.path.join(os.path.dirname(__file__), "session_log.csv")
ANALYZE_EVERY_N_FRAMES = 5  # DeepFace e pesado; nao roda a cada frame
CALIBRATION_SECONDS = 12
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

LOG_FIELDS = [
    "timestamp", "dominant_emotion", *EMOTIONS,
    "blink_rate_min", "gaze_offset", "hr_bpm",
    "shoulder_tilt_deg", "lean_proxy", "incongruence_score",
]


def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)


def write_log_header_now():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LOG_FIELDS)


def log_row(dominant_emotion, scores, extra):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        row = (
            [datetime.now().isoformat(timespec="seconds"), dominant_emotion]
            + [round(scores.get(e, 0.0), 2) for e in EMOTIONS]
            + [
                _fmt(extra.get("blink_rate_min")),
                _fmt(extra.get("gaze_offset")),
                _fmt(extra.get("hr_bpm")),
                _fmt(extra.get("shoulder_tilt_deg")),
                _fmt(extra.get("lean_proxy")),
                _fmt(extra.get("incongruence_score")),
            ]
        )
        csv.writer(f).writerow(row)


def _fmt(v):
    return "" if v is None else round(v, 3)


TEXT_DARK = (20, 20, 20)
PANEL_BG = (235, 235, 235)


def stress_label(score):
    if score is None:
        return "sem baseline (aperte 'b')", (90, 90, 90)
    if score < 1.0:
        return "baixo", (0, 130, 0)
    if score < 2.0:
        return "medio", (0, 140, 200)
    return "alto", (0, 0, 200)


def draw_overlay(frame, region, dominant_emotion, scores, logging_on, extra, calibrating, calib_remaining):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 130, 0), 2)

    label = EMOTION_LABELS_PT.get(dominant_emotion, dominant_emotion)
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    label_y = max(y - 10, label_h + 10)
    cv2.rectangle(
        frame, (x - 2, label_y - label_h - 8), (x + label_w + 6, label_y + 4),
        PANEL_BG, -1,
    )
    cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_DARK, 2)

    panel_w = 320
    panel_h = 22 * len(EMOTIONS) + 40
    cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), PANEL_BG, -1)

    bar_x, bar_y = 15, 35
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

    # --- painel de sinais complementares (canto superior direito) ---
    if not SIGNALS_AVAILABLE:
        return

    fh, fw = frame.shape[:2]
    panel2_w, panel2_h = 300, 120
    px0 = fw - panel2_w - 5
    cv2.rectangle(frame, (px0, 5), (fw - 5, 5 + panel2_h), PANEL_BG, -1)

    ty = 28
    if calibrating:
        cv2.putText(
            frame, f"Calibrando baseline... {calib_remaining:.0f}s",
            (px0 + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2,
        )
        cv2.putText(
            frame, "Fique parado, expressao neutra",
            (px0 + 10, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DARK, 1,
        )
        return

    def line(text, dy):
        cv2.putText(frame, text, (px0 + 10, ty + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DARK, 1)

    hr = extra.get("hr_bpm")
    line(f"BPM (aprox.): {hr:.0f}" if hr else "BPM (aprox.): --", 0)
    blink = extra.get("blink_rate_min")
    line(f"Piscadas/min: {blink:.0f}" if blink is not None else "Piscadas/min: --", 22)
    gaze = extra.get("gaze_offset")
    line(f"Desvio de olhar: {gaze:.2f}" if gaze is not None else "Desvio de olhar: --", 44)
    tilt = extra.get("shoulder_tilt_deg")
    line(f"Inclinacao ombros: {tilt:.1f}graus" if tilt is not None else "Inclinacao ombros: --", 66)

    score = extra.get("incongruence_score")
    txt, color = stress_label(score)
    line(f"Sinal de estresse: {txt}", 90)


def compute_incongruence(calibrator, **signals):
    if not calibrator or not calibrator.calibrated:
        return None
    zscores = calibrator.zscore(**signals)
    if not zscores:
        return None
    clipped = [min(abs(v), 4.0) for v in zscores.values()]
    return sum(clipped) / len(clipped)


def main():
    ensure_log_header()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 1 or fps > 120:
        fps = 30

    face_mesh = pose_tracker = rppg = calibrator = None
    if SIGNALS_AVAILABLE:
        try:
            face_mesh = FaceMeshTracker()
            pose_tracker = PoseTracker()
            rppg = RPPGEstimator(fps=fps)
            calibrator = BaselineCalibrator()
        except ImportError as e:
            print(f"[aviso] nao foi possivel iniciar sinais complementares: {e}")

    frame_count = 0
    last_region = None
    last_dominant = "neutral"
    last_scores = {e: 0.0 for e in EMOTIONS}
    logging_on = False

    calibrating = False
    calib_end_time = 0.0

    emotion_history = collections.deque(maxlen=200)  # (timestamp, dominant_emotion)
    extra = {
        "blink_rate_min": None, "gaze_offset": None, "hr_bpm": None,
        "shoulder_tilt_deg": None, "lean_proxy": None, "incongruence_score": None,
    }

    print("Pressione 'q' sair | 'l' liga/desliga log | 'c' limpa log | 'b' calibrar baseline")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()

        # --- sinais complementares (todo frame, sao leves) ---
        gaze_offset = shoulder_tilt = lean_proxy = None
        if face_mesh is not None:
            fm = face_mesh.process(frame)
            if fm is not None:
                gaze_offset = fm["gaze_offset"]
                if rppg is not None:
                    rppg.add_frame(frame, fm["forehead_box"])
                extra["blink_rate_min"] = face_mesh.blink_rate_per_min()

        if pose_tracker is not None:
            pr = pose_tracker.process(frame)
            if pr is not None:
                shoulder_tilt = pr["shoulder_tilt_deg"]
                lean_proxy = pr["lean_proxy"]

        hr_bpm = rppg.estimate_bpm() if rppg is not None else None
        extra["gaze_offset"] = gaze_offset
        extra["hr_bpm"] = hr_bpm
        extra["shoulder_tilt_deg"] = shoulder_tilt
        extra["lean_proxy"] = lean_proxy

        if calibrating:
            extra["incongruence_score"] = None
            calibrator.add_sample(
                gaze_offset=gaze_offset, hr_bpm=hr_bpm,
                shoulder_tilt_deg=shoulder_tilt, lean_proxy=lean_proxy,
                blink_rate_min=extra["blink_rate_min"],
            )
            if now >= calib_end_time:
                calibrating = False
                ok_calib = calibrator.finalize()
                print("Baseline calibrado." if ok_calib else "Calibracao insuficiente, tente 'b' de novo.")
        else:
            extra["incongruence_score"] = compute_incongruence(
                calibrator,
                gaze_offset=gaze_offset, hr_bpm=hr_bpm,
                shoulder_tilt_deg=shoulder_tilt, lean_proxy=lean_proxy,
                blink_rate_min=extra["blink_rate_min"],
            )

        # --- emocao (DeepFace, mais pesado, roda a cada N frames) ---
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
                emotion_history.append((now, last_dominant))

                if logging_on and last_region["w"] > 0:
                    log_row(last_dominant, last_scores, extra)
            except Exception as e:
                print(f"[aviso] falha na analise: {e}")

        if last_region and last_region.get("w", 0) > 0:
            calib_remaining = max(0.0, calib_end_time - now)
            draw_overlay(
                frame, last_region, last_dominant, last_scores, logging_on,
                extra, calibrating, calib_remaining,
            )
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
            write_log_header_now()
            print("Log limpo.")
        elif key == ord("b"):
            if calibrator is None:
                print("Sinais complementares indisponiveis - instale mediapipe/scipy.")
            else:
                calibrator.reset()
                calibrating = True
                calib_end_time = time.time() + CALIBRATION_SECONDS
                print(f"Calibrando por {CALIBRATION_SECONDS}s - fique parado, expressao neutra.")

    cap.release()
    cv2.destroyAllWindows()
    if face_mesh is not None:
        face_mesh.close()
    if pose_tracker is not None:
        pose_tracker.close()


if __name__ == "__main__":
    main()
