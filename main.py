"""
Analise de expressao facial em tempo real - uso individual e local.

Este script roda 100% na sua maquina: nenhuma imagem, audio ou dado sai
do computador. O objetivo e autoconhecimento (ver seu proprio padrao de
humor/estresse ao longo de uma sessao), nao vigilancia, ranking de
terceiros ou deteccao de mentira. Veja o README para o porque disso.

Sinais capturados:
  - Emocao dominante (DeepFace): macro-expressoes.
  - Piscadas e desvio de olhar (MediaPipe Face Mesh).
  - Frequencia cardiaca aproximada via rPPG (MediaPipe + testa), com
    checagem de qualidade (luz/movimento) - BPM some da tela quando a
    leitura nao e confiavel em vez de mostrar um numero enganoso.
  - Postura: inclinacao dos ombros e proxy de inclinar o tronco
    (MediaPipe Pose).
  - Voz: volume, taxa de pausas na fala e pitch aproximado (microfone).
    Audio bruto nunca e salvo, so os valores derivados.
  - Um "score de incongruencia/estresse", que so existe DEPOIS de
    calibrar seu proprio baseline neutro (tecla 'b'). Ele mede o quanto
    voce se afastou do SEU proprio normal - nao indica mentira.

Todos os sinais numericos passam por uma media movel exponencial (EMA)
antes de aparecer na tela/log, pra reduzir tremulacao sem esconder
tendencia real.

A calibracao do baseline ('b') tambem ajusta automaticamente a
sensibilidade de piscada (EAR) e a faixa de luz aceitavel do BPM pro seu
rosto/ambiente, em vez de usar limiares fixos genericos.

Controles:
  q - sair (mostra um resumo da sessao no terminal)
  l - liga/desliga o log da sessao (grava em session_log.csv)
  c - arquiva o log atual (session_archive/) e comeca um novo
  b - (re)inicia a calibracao do baseline (fique parado, expressao neutra)
  n - anota/troca a nota de contexto da sessao atual (ex.: "reuniao X")

Uso:
  python main.py                     # camera/microfone padrao do sistema
  python main.py --camera 1          # usa a webcam de indice 1
  python main.py --mic 2             # usa o microfone de indice 2
  python main.py --note "reuniao X"  # marca a sessao com uma nota
  python main.py --list-devices      # lista cameras/microfones e sai
"""

import argparse
import collections
import csv
import os
import sys
import time
from datetime import datetime

import cv2
from deepface import DeepFace

try:
    from signals import (
        BaselineCalibrator, EMA, FaceMeshTracker, PoseTracker, RPPGEstimator, VoiceTracker,
    )
    SIGNALS_AVAILABLE = True
except ImportError as e:
    print(f"[aviso] sinais complementares desativados ({e}). "
          f"Rode: pip install -r requirements.txt")
    SIGNALS_AVAILABLE = False

# Rodando como .exe (PyInstaller), __file__ aponta pra dentro do bundle,
# nao pra pasta ao lado do executavel - usa sys.executable nesse caso, pra
# session_log.csv/session_archive ficarem num lugar previsivel e gravavel.
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "session_log.csv")
ARCHIVE_DIR = os.path.join(BASE_DIR, "session_archive")
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

# chaves dos sinais complementares que entram no baseline/z-score - uma
# unica lista pra nao duplicar (e desalinhar) entre calibracao e leitura
SIGNAL_KEYS = [
    "gaze_offset", "hr_bpm", "shoulder_tilt_deg", "lean_proxy", "blink_rate_min",
    "voice_rms", "voice_pitch_hz", "voice_pause_rate_min",
]

LOG_FIELDS = [
    "timestamp", "dominant_emotion", *EMOTIONS,
    "blink_rate_min", "gaze_offset", "hr_bpm",
    "shoulder_tilt_deg", "lean_proxy",
    "voice_rms", "voice_pitch_hz", "voice_pause_rate_min",
    "incongruence_score", "session_note",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analise de expressao facial e sinais complementares - uso individual e local.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam (padrao: 0).")
    parser.add_argument(
        "--mic", type=int, default=None,
        help="Indice do microfone (padrao: dispositivo de entrada padrao do sistema).",
    )
    parser.add_argument(
        "--note", type=str, default="", metavar="TEXTO",
        help="Nota curta de contexto pra essa sessao (ex.: 'reuniao X'), gravada em cada linha do log.",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Lista as cameras e microfones disponiveis e sai, sem abrir a janela.",
    )
    return parser.parse_args()


def list_devices():
    print("Cameras disponiveis (indice: resolucao):")
    found_camera = False
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {i}: {w}x{h}")
            found_camera = True
        cap.release()
    if not found_camera:
        print("  nenhuma encontrada nos indices 0-5.")

    print("\nMicrofones disponiveis (indice: nome):")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        found_mic = False
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"  {idx}: {dev['name']}")
                found_mic = True
        if not found_mic:
            print("  nenhum microfone encontrado.")
    except Exception as e:
        print(f"  nao foi possivel listar microfones ({e}). Rode: pip install -r requirements.txt")


def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)


def write_log_header_now():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LOG_FIELDS)


def archive_current_log():
    """Move o session_log.csv atual pra session_archive/ com timestamp no
    nome, em vez de destruir os dados - assim trends.py consegue comparar
    sessoes antigas. So arquiva se tiver dado de verdade (mais que so o
    cabecalho)."""
    if not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH, encoding="utf-8") as f:
        has_data = len(f.readlines()) > 1
    if not has_data:
        return None
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(ARCHIVE_DIR, f"session_{stamp}.csv")
    os.replace(LOG_PATH, dest)
    return dest


def log_row(dominant_emotion, scores, extra, session_note=""):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        row = (
            [datetime.now().isoformat(timespec="seconds"), dominant_emotion]
            + [round(scores.get(e, 0.0), 2) for e in EMOTIONS]
            + [_fmt(extra.get(k)) for k in (
                "blink_rate_min", "gaze_offset", "hr_bpm",
                "shoulder_tilt_deg", "lean_proxy",
                "voice_rms", "voice_pitch_hz", "voice_pause_rate_min",
                "incongruence_score",
            )]
            + [session_note]
        )
        csv.writer(f).writerow(row)


def _fmt(v):
    return "" if v is None else round(v, 3)


TEXT_DARK = (20, 20, 20)
PANEL_BG = (235, 235, 235)


def stress_label(score):
    if score is None:
        return "sem baseline", (90, 90, 90)
    if score < 1.0:
        return "baixo", (0, 130, 0)
    if score < 2.0:
        return "medio", (0, 140, 200)
    return "alto", (0, 0, 200)


def draw_overlay(frame, region, dominant_emotion, scores, logging_on, extra, calibrating, calib_remaining, session_note=""):
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

    if session_note:
        cv2.putText(
            frame, f"Nota: {session_note}", (15, bar_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 60, 0), 1,
        )

    # --- painel de sinais complementares (canto superior direito) ---
    if not SIGNALS_AVAILABLE:
        return

    fh, fw = frame.shape[:2]
    panel2_w, panel2_h = 350, 170
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
        cv2.putText(frame, text, (px0 + 10, ty + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_DARK, 1)

    hr = extra.get("hr_bpm")
    hr_note = extra.get("hr_quality_note")
    if hr is not None:
        line(f"BPM (aprox.): {hr:.0f}", 0)
    else:
        line(f"BPM (aprox.): -- ({hr_note})" if hr_note else "BPM (aprox.): --", 0)

    blink = extra.get("blink_rate_min")
    line(f"Piscadas/min: {blink:.0f}" if blink is not None else "Piscadas/min: --", 21)
    gaze = extra.get("gaze_offset")
    line(f"Desvio de olhar: {gaze:.2f}" if gaze is not None else "Desvio de olhar: --", 42)
    tilt = extra.get("shoulder_tilt_deg")
    line(f"Inclinacao ombros: {tilt:.1f}graus" if tilt is not None else "Inclinacao ombros: --", 63)

    pause = extra.get("voice_pause_rate_min")
    line(f"Pausas na fala/min: {pause:.0f}" if pause is not None else "Pausas na fala/min: --", 84)
    pitch = extra.get("voice_pitch_hz")
    line(f"Pitch da voz (aprox.): {pitch:.0f}Hz" if pitch is not None else "Pitch da voz (aprox.): --", 105)

    score = extra.get("incongruence_score")
    txt, color = stress_label(score)
    line(f"Sinal de estresse: {txt}", 130)


def compute_incongruence(calibrator, **signals):
    if not calibrator or not calibrator.calibrated:
        return None
    zscores = calibrator.zscore(**signals)
    if not zscores:
        return None
    clipped = [min(abs(v), 4.0) for v in zscores.values()]
    return sum(clipped) / len(clipped)


def print_session_summary(start_time, emotion_counts, stress_counts):
    duration_s = time.time() - start_time
    mins, secs = divmod(int(duration_s), 60)

    print("\n--- Resumo da sessao ---")
    print(f"Duracao: {mins}min {secs}s")

    total_e = sum(emotion_counts.values())
    if total_e:
        print("Emocao dominante (por amostra do DeepFace):")
        for emo, cnt in emotion_counts.most_common():
            print(f"  {EMOTION_LABELS_PT.get(emo, emo)}: {100 * cnt / total_e:.0f}%")

    total_s = sum(stress_counts.values())
    if total_s:
        print("Sinal de estresse (apos calibrar baseline com 'b'):")
        for lbl in ("baixo", "medio", "alto"):
            if lbl in stress_counts:
                print(f"  {lbl}: {100 * stress_counts[lbl] / total_s:.0f}% do tempo com baseline calibrado")
    else:
        print("Baseline nao calibrado nesta sessao - aperte 'b' na proxima pra ter esse dado.")
    print("Lembrete: isto e sinal de estresse/incongruencia pra autoconhecimento, nao deteccao de mentira.")
    print("------------------------\n")


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    session_note = args.note or ""

    ensure_log_header()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir a webcam de indice {args.camera}. "
            f"Rode com --list-devices pra ver o que esta disponivel."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 1 or fps > 120:
        fps = 30

    calibrator = BaselineCalibrator() if SIGNALS_AVAILABLE else None

    face_mesh = pose_tracker = rppg = voice = None
    if SIGNALS_AVAILABLE:
        try:
            face_mesh = FaceMeshTracker()
        except ImportError as e:
            print(f"[aviso] piscadas/desvio de olhar desativados: {e}")
        try:
            pose_tracker = PoseTracker()
        except ImportError as e:
            print(f"[aviso] postura desativada: {e}")
        try:
            rppg = RPPGEstimator(fps=fps)
        except ImportError as e:
            print(f"[aviso] BPM (rPPG) desativado: {e}")
        try:
            voice = VoiceTracker(device=args.mic)
            voice.start()
        except Exception as e:
            print(f"[aviso] sinal de voz desativado ({e}). Verifique se o microfone esta disponivel/permitido "
                  f"(ou tente --list-devices pra escolher outro).")
            voice = None

    # suavizacao (EMA) - reduz tremulacao dos sinais na tela/log
    gaze_ema = EMA(alpha=0.3) if SIGNALS_AVAILABLE else None
    tilt_ema = EMA(alpha=0.3) if SIGNALS_AVAILABLE else None
    hr_ema = EMA(alpha=0.2) if SIGNALS_AVAILABLE else None
    pitch_ema = EMA(alpha=0.3) if SIGNALS_AVAILABLE else None
    stress_ema = EMA(alpha=0.25) if SIGNALS_AVAILABLE else None

    frame_count = 0
    last_region = None
    last_dominant = "neutral"
    last_scores = {e: 0.0 for e in EMOTIONS}
    logging_on = False

    calibrating = False
    calib_end_time = 0.0
    ear_samples = []
    brightness_samples = []

    extra = {k: None for k in (*SIGNAL_KEYS, "incongruence_score", "hr_quality_note")}

    session_start = time.time()
    emotion_counts = collections.Counter()
    stress_counts = collections.Counter()

    print("Pressione 'q' sair | 'l' liga/desliga log | 'c' arquiva+limpa log | 'b' calibrar baseline | 'n' anotar sessao")
    if session_note:
        print(f"Nota da sessao: {session_note}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()

            # --- sinais complementares (todo frame, sao leves) ---
            gaze_raw = shoulder_tilt_raw = lean_proxy = ear_raw = None
            if face_mesh is not None:
                fm = face_mesh.process(frame)
                if fm is not None:
                    gaze_raw = fm["gaze_offset"]
                    ear_raw = fm["ear"]
                    if rppg is not None:
                        rppg.add_frame(frame, fm["forehead_box"])
                    extra["blink_rate_min"] = face_mesh.blink_rate_per_min()

            if pose_tracker is not None:
                pr = pose_tracker.process(frame)
                if pr is not None:
                    shoulder_tilt_raw = pr["shoulder_tilt_deg"]
                    lean_proxy = pr["lean_proxy"]

            hr_reliable = False
            hr_raw = None
            if rppg is not None:
                hr_raw = rppg.estimate_bpm()
                hr_reliable = rppg.is_reliable()
                if not hr_reliable:
                    extra["hr_quality_note"] = (
                        "pouca luz" if not rppg.lighting_ok else
                        "movimento" if not rppg.motion_ok else
                        "sinal fraco"
                    )
                else:
                    extra["hr_quality_note"] = None

            if voice is not None:
                vread = voice.read()
                extra["voice_rms"] = vread.get("rms")
                if pitch_ema is not None:
                    pitch_ema.update(vread.get("pitch_hz"))
                    extra["voice_pitch_hz"] = pitch_ema.value
                else:
                    extra["voice_pitch_hz"] = vread.get("pitch_hz")
                extra["voice_pause_rate_min"] = voice.pause_rate_per_min()

            extra["gaze_offset"] = gaze_ema.update(gaze_raw) if gaze_ema else gaze_raw
            extra["shoulder_tilt_deg"] = tilt_ema.update(shoulder_tilt_raw) if tilt_ema else shoulder_tilt_raw
            extra["lean_proxy"] = lean_proxy
            extra["hr_bpm"] = (hr_ema.update(hr_raw) if hr_ema else hr_raw) if (hr_raw is not None and hr_reliable) else None

            sample_kwargs = {k: extra.get(k) for k in SIGNAL_KEYS}

            if calibrating:
                extra["incongruence_score"] = None
                if stress_ema:
                    stress_ema.reset()
                calibrator.add_sample(**sample_kwargs)
                if ear_raw is not None:
                    ear_samples.append(ear_raw)
                if rppg is not None and rppg.last_brightness is not None:
                    brightness_samples.append(rppg.last_brightness)
                if now >= calib_end_time:
                    calibrating = False
                    ok_calib = calibrator.finalize()
                    tuned = []
                    if face_mesh is not None and face_mesh.calibrate_blink_threshold(ear_samples):
                        tuned.append("piscada")
                    if rppg is not None and rppg.calibrate_lighting(brightness_samples):
                        tuned.append("luz do BPM")
                    ear_samples = []
                    brightness_samples = []
                    if ok_calib:
                        extra_msg = f" (sensibilidade ajustada: {', '.join(tuned)})" if tuned else ""
                        print(f"Baseline calibrado.{extra_msg}")
                    else:
                        print("Calibracao insuficiente, tente 'b' de novo.")
            else:
                raw_score = compute_incongruence(calibrator, **sample_kwargs)
                extra["incongruence_score"] = stress_ema.update(raw_score) if (stress_ema and raw_score is not None) else raw_score
                if extra["incongruence_score"] is not None:
                    lbl, _ = stress_label(extra["incongruence_score"])
                    stress_counts[lbl] += 1

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
                    emotion_counts[last_dominant] += 1

                    if logging_on and last_region["w"] > 0:
                        log_row(last_dominant, last_scores, extra, session_note)
                except Exception as e:
                    print(f"[aviso] falha na analise: {e}")

            if last_region and last_region.get("w", 0) > 0:
                calib_remaining = max(0.0, calib_end_time - now)
                draw_overlay(
                    frame, last_region, last_dominant, last_scores, logging_on,
                    extra, calibrating, calib_remaining, session_note,
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
                dest = archive_current_log()
                write_log_header_now()
                if dest:
                    print(f"Sessao anterior arquivada em: {dest}")
                print("Log limpo (pronto pra nova sessao). Use trends.py pra comparar sessoes arquivadas.")
            elif key == ord("b"):
                if calibrator is None:
                    print("Sinais complementares indisponiveis - instale mediapipe/scipy.")
                else:
                    calibrator.reset()
                    ear_samples = []
                    brightness_samples = []
                    calibrating = True
                    calib_end_time = time.time() + CALIBRATION_SECONDS
                    print(f"Calibrando por {CALIBRATION_SECONDS}s - fique parado, expressao neutra.")
            elif key == ord("n"):
                # janela do OpenCV congela um instante enquanto espera o
                # texto no terminal - aceitavel, so acontece quando pedido
                print(f"Nota atual: '{session_note}'. Digite a nova nota e Enter (vazio cancela):")
                try:
                    new_note = input("> ").strip()
                except EOFError:
                    new_note = ""
                if new_note:
                    session_note = new_note
                    print(f"Nota atualizada: {session_note}")
                else:
                    print("Nota mantida sem alteracao.")
    except KeyboardInterrupt:
        print("\n[aviso] Interrompido com Ctrl+C - prefira apertar 'q' com a janela em foco pra sair.")
    finally:
        # um segundo Ctrl+C durante a propria limpeza nao pode derrubar o
        # resumo da sessao nem impedir os outros recursos de fechar -
        # cada passo e isolado e best-effort
        for cleanup in (
            cap.release,
            cv2.destroyAllWindows,
            face_mesh.close if face_mesh is not None else None,
            pose_tracker.close if pose_tracker is not None else None,
            voice.close if voice is not None else None,
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException:
                pass
        print_session_summary(session_start, emotion_counts, stress_counts)


if __name__ == "__main__":
    main()
