"""
Sinais complementares de fisiologia e postura via webcam.

IMPORTANTE: nada aqui detecta mentira. O que este modulo faz e extrair
sinais (frequencia cardiaca aproximada, piscadas, desvio de olhar,
postura) que, comparados com o PROPRIO baseline neutro da pessoa,
apontam momentos de possivel estresse ou incongruencia emocional para
REVISAO HUMANA - nunca um veredito automatico de verdade/mentira. Veja o
README para o embasamento cientifico dessa limitacao.

Referencias de metodo:
- rPPG: estimativa de frequencia cardiaca a partir da variacao sutil de
  cor da pele na testa, via filtragem passa-banda + FFT. Precisao cai
  muito com movimento da cabeca e mudanca de iluminacao - tratar como
  tendencia aproximada, nao como valor clinico.
- EAR (Eye Aspect Ratio): razao geometrica dos pontos do olho, usada para
  detectar piscadas (Soukupova & Cech, 2016), adaptada aos indices do
  MediaPipe Face Mesh.
- Gaze: posicao da iris relativa aos cantos do olho, usando os pontos de
  iris do MediaPipe Face Mesh (refine_landmarks=True).
- Postura: angulo de inclinacao dos ombros e distancia ombro-nariz (proxy
  de inclinar o tronco pra frente/tras), via MediaPipe Pose.
- Voz: volume (RMS), taxa de pausas na fala e pitch aproximado por
  autocorrelacao, via microfone (sounddevice). Audio bruto nunca e salvo.
"""

import collections
import threading
import time

import numpy as np

try:
    import mediapipe as mp
    _MP_IMPORT_ERROR = None
except Exception as e:  # nao so ImportError: no Windows, DLL/protobuf
    # quebrado costuma estourar ImportError/RuntimeError/OSError diferentes
    mp = None
    _MP_IMPORT_ERROR = e

try:
    from scipy.signal import butter, filtfilt
    _SCIPY_IMPORT_ERROR = None
except Exception as e:
    butter = filtfilt = None
    _SCIPY_IMPORT_ERROR = e

try:
    import sounddevice as sd
    _SD_IMPORT_ERROR = None
except Exception as e:
    sd = None
    _SD_IMPORT_ERROR = e


class EMA:
    """Media movel exponencial simples, usada pra suavizar sinais
    ruidosos (gaze, BPM, inclinacao dos ombros, score de incongruencia)
    antes de exibir/logar - reduz flicker sem esconder tendencia real."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if x is None:
            return self.value
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# --- Indices do MediaPipe Face Mesh (468 pontos + 10 de iris quando
# refine_landmarks=True). Conferir contra a versao instalada do mediapipe
# se o comportamento parecer estranho - esses indices sao os comumente
# documentados pela propria equipe do MediaPipe para EAR/iris. ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
FOREHEAD_REGION = [10, 338, 297, 332, 284, 251, 54, 21, 68, 103]


def _require_mediapipe():
    if mp is None:
        raise ImportError(
            f"mediapipe nao pode ser importado ({_MP_IMPORT_ERROR!r}). "
            f"Se 'pip show mediapipe' confirma que esta instalado, o problema "
            f"provavelmente e o Microsoft Visual C++ Redistributable x64 "
            f"faltando/desatualizado no Windows (DLL load failed), ou um "
            f"conflito de versao com outra dependencia (numpy/protobuf)."
        )


def _eye_aspect_ratio(pts, idx):
    p = pts[idx]
    vert1 = np.linalg.norm(p[1] - p[5])
    vert2 = np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


class FaceMeshTracker:
    """Extrai piscadas e desvio de olhar (gaze) por frame, mais a caixa
    da testa (usada pelo RPPGEstimator)."""

    EAR_BLINK_THRESHOLD = 0.21
    EAR_CONSEC_FRAMES = 2

    def __init__(self):
        _require_mediapipe()
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._closed_frames = 0
        self.blink_count = 0
        self.blink_timestamps = collections.deque(maxlen=300)

    def process(self, frame_bgr):
        """Retorna dict com ear, gaze_offset, blinked_now, forehead_box,
        ou None se nenhum rosto foi encontrado."""
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm])

        ear = (_eye_aspect_ratio(pts, LEFT_EYE) + _eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0

        blinked_now = False
        if ear < self.EAR_BLINK_THRESHOLD:
            self._closed_frames += 1
        else:
            if self._closed_frames >= self.EAR_CONSEC_FRAMES:
                blinked_now = True
                self.blink_count += 1
                self.blink_timestamps.append(time.time())
            self._closed_frames = 0

        gaze_offset = self._gaze_offset(pts)

        forehead_pts = pts[FOREHEAD_REGION]
        x0, y0 = forehead_pts.min(axis=0)
        x1, y1 = forehead_pts.max(axis=0)
        forehead_box = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        return {
            "ear": ear,
            "blinked_now": blinked_now,
            "gaze_offset": gaze_offset,
            "forehead_box": forehead_box,
        }

    def _gaze_offset(self, pts):
        """~0 = olhando pro centro do proprio olho; valores maiores =
        iris desviada pra lateral (proxy grosseiro de desviar o olhar)."""
        offsets = []
        for iris_idx, corners in (
            (LEFT_IRIS, LEFT_EYE_CORNERS),
            (RIGHT_IRIS, RIGHT_EYE_CORNERS),
        ):
            iris_center = pts[iris_idx].mean(axis=0)
            c0, c1 = pts[corners[0]], pts[corners[1]]
            eye_center = (c0 + c1) / 2.0
            eye_width = np.linalg.norm(c1 - c0)
            if eye_width == 0:
                continue
            offsets.append(np.linalg.norm(iris_center - eye_center) / eye_width)
        return float(np.mean(offsets)) if offsets else 0.0

    def blink_rate_per_min(self, window_s=60):
        now = time.time()
        recent = [t for t in self.blink_timestamps if now - t <= window_s]
        return len(recent) * (60.0 / window_s)

    def close(self):
        self._mesh.close()


class RPPGEstimator:
    """
    Estimativa aproximada de frequencia cardiaca (BPM) via variacao de
    cor da pele na testa (rPPG, canal verde). Sensivel a movimento e
    iluminacao - tratar como tendencia, nao como valor clinico. Nunca
    substitui um oximetro/monitor cardiaco de verdade.
    """

    # limiares empiricos, grosseiros de proposito - o objetivo e so
    # descartar leituras obviamente ruins (escuro demais, estourado de
    # luz, cabeca se mexendo muito), nao validar precisao clinica
    MIN_BRIGHTNESS = 25
    MAX_BRIGHTNESS = 230
    MAX_MOTION_RATIO = 0.15  # deslocamento do centro da testa vs largura dela
    MIN_QUALITY = 0.15  # fracao da energia do espectro concentrada no pico

    def __init__(self, fps=30, buffer_seconds=8, hr_min=42, hr_max=180):
        if butter is None:
            raise ImportError(
                f"scipy nao pode ser importado ({_SCIPY_IMPORT_ERROR!r}). "
                f"Rode: pip install -r requirements.txt"
            )
        self.fps = fps
        self.buffer_len = int(fps * buffer_seconds)
        self.buffer = collections.deque(maxlen=self.buffer_len)
        self.hr_min = hr_min
        self.hr_max = hr_max
        self._last_bpm = None
        self._last_quality = 0.0
        self._last_box_center = None
        self.lighting_ok = True
        self.motion_ok = True

    def add_frame(self, frame_bgr, forehead_box):
        x, y, w, h = forehead_box
        if w <= 0 or h <= 0:
            self.motion_ok = False
            return
        x, y = max(x, 0), max(y, 0)
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size == 0:
            return

        center = (x + w / 2.0, y + h / 2.0)
        if self._last_box_center is not None:
            disp = np.hypot(center[0] - self._last_box_center[0], center[1] - self._last_box_center[1])
            self.motion_ok = disp < (w * self.MAX_MOTION_RATIO)
        self._last_box_center = center

        brightness = float(np.mean(roi))
        self.lighting_ok = self.MIN_BRIGHTNESS < brightness < self.MAX_BRIGHTNESS

        # canal verde tem melhor relacao sinal/ruido para variacao de
        # volume sanguineo na pele
        self.buffer.append(float(np.mean(roi[:, :, 1])))

    def estimate_bpm(self):
        if len(self.buffer) < self.buffer_len:
            return self._last_bpm

        signal = np.array(self.buffer, dtype=np.float64)
        signal = signal - np.mean(signal)

        nyq = 0.5 * self.fps
        low = (self.hr_min / 60.0) / nyq
        high = (self.hr_max / 60.0) / nyq
        try:
            b, a = butter(3, [low, high], btype="band")
            filtered = filtfilt(b, a, signal)
        except ValueError:
            return self._last_bpm

        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.fps)
        power = np.abs(np.fft.rfft(filtered)) ** 2

        band = (freqs >= self.hr_min / 60.0) & (freqs <= self.hr_max / 60.0)
        if not np.any(band):
            return self._last_bpm

        band_power = power[band]
        peak_idx = int(np.argmax(band_power))
        peak_freq = freqs[band][peak_idx]
        total = float(np.sum(band_power)) + 1e-9
        self._last_quality = float(band_power[peak_idx]) / total

        self._last_bpm = float(peak_freq * 60.0)
        return self._last_bpm

    def is_reliable(self):
        """False = leitura de BPM nao deve ser mostrada/usada no baseline
        (pouca luz, muito movimento, ou pico espectral fraco/ambiguo)."""
        return self.lighting_ok and self.motion_ok and self._last_quality >= self.MIN_QUALITY


class PoseTracker:
    """
    Linguagem corporal complementar via MediaPipe Pose: inclinacao dos
    ombros (assimetria) e proxy de inclinar o tronco pra frente/tras.
    Sinal grosseiro - util como mais um ponto de dado, nao isoladamente.
    """

    def __init__(self):
        _require_mediapipe()
        self._pose = mp.solutions.pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        L_SH, R_SH, NOSE = 11, 12, 0

        left = np.array([lm[L_SH].x * w, lm[L_SH].y * h])
        right = np.array([lm[R_SH].x * w, lm[R_SH].y * h])
        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])

        shoulder_vec = right - left
        shoulder_tilt_deg = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))

        shoulder_mid = (left + right) / 2.0
        shoulder_width = np.linalg.norm(shoulder_vec) or 1.0
        # distancia nariz->linha dos ombros normalizada pela largura dos
        # ombros: proxy grosseiro de inclinar o tronco pra frente (valor
        # cai) ou se afastar da camera (valor sobe)
        lean_proxy = float(np.linalg.norm(nose - shoulder_mid) / shoulder_width)

        return {
            "shoulder_tilt_deg": shoulder_tilt_deg,
            "lean_proxy": lean_proxy,
            "left_shoulder": tuple(left.astype(int)),
            "right_shoulder": tuple(right.astype(int)),
        }

    def close(self):
        self._pose.close()


class VoiceTracker:
    """
    Sinal de voz complementar via microfone: volume (RMS), taxa de pausas
    na fala e um pitch aproximado (autocorrelacao simples - nao e um
    detector de pitch de qualidade musical, so uma tendencia). Roda a
    captura numa thread separada (callback do sounddevice); process()
    so le o ultimo estado calculado, sem bloquear o loop de video.

    Audio bruto NUNCA e salvo em disco - so os valores derivados (rms,
    pitch, pausa) entram no log, do mesmo jeito que os outros sinais.
    """

    SILENCE_RMS = 0.01
    PITCH_MIN_HZ = 70
    PITCH_MAX_HZ = 400

    def __init__(self, samplerate=16000, block_seconds=0.5):
        if sd is None:
            raise ImportError(
                f"sounddevice nao pode ser importado ({_SD_IMPORT_ERROR!r}). "
                f"Rode: pip install -r requirements.txt"
            )
        self.samplerate = samplerate
        self.block_size = int(samplerate * block_seconds)

        self._lock = threading.Lock()
        self._latest = {"rms": None, "pitch_hz": None, "is_silent": True}
        self._pause_timestamps = collections.deque(maxlen=300)
        self._was_silent = True
        self._stream = None

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        audio = np.asarray(indata[:, 0], dtype=np.float64)
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        is_silent = rms < self.SILENCE_RMS

        pitch_hz = None if is_silent else self._estimate_pitch(audio)

        with self._lock:
            self._latest = {"rms": rms, "pitch_hz": pitch_hz, "is_silent": is_silent}
            if is_silent and not self._was_silent:
                self._pause_timestamps.append(time.time())
            self._was_silent = is_silent

    def _estimate_pitch(self, audio):
        audio = audio - np.mean(audio)
        corr = np.correlate(audio, audio, mode="full")
        corr = corr[len(corr) // 2:]

        min_lag = int(self.samplerate / self.PITCH_MAX_HZ)
        max_lag = int(self.samplerate / self.PITCH_MIN_HZ)
        if max_lag >= len(corr) or min_lag >= max_lag:
            return None

        segment = corr[min_lag:max_lag]
        if segment.size == 0 or np.max(segment) <= 0:
            return None

        peak_lag = min_lag + int(np.argmax(segment))
        if peak_lag == 0:
            return None
        return float(self.samplerate / peak_lag)

    def read(self):
        with self._lock:
            return dict(self._latest)

    def pause_rate_per_min(self, window_s=60):
        now = time.time()
        recent = [t for t in self._pause_timestamps if now - t <= window_s]
        return len(recent) * (60.0 / window_s)

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class BaselineCalibrator:
    """
    Guarda media/desvio-padrao de cada sinal durante uma fase de
    calibracao (pessoa em estado neutro, ~10s) e depois calcula o
    z-score de novas leituras contra o PROPRIO baseline da pessoa -
    nunca contra uma media populacional ou "normal" generico.
    """

    MIN_SAMPLES = 5

    def __init__(self):
        self._samples = collections.defaultdict(list)
        self._mean = {}
        self._std = {}
        self.calibrated = False

    def add_sample(self, **signals):
        for k, v in signals.items():
            if v is not None:
                self._samples[k].append(v)

    def finalize(self):
        self._mean.clear()
        self._std.clear()
        for k, vals in self._samples.items():
            if len(vals) >= self.MIN_SAMPLES:
                self._mean[k] = float(np.mean(vals))
                self._std[k] = float(np.std(vals)) or 1.0
        self.calibrated = bool(self._mean)
        return self.calibrated

    def zscore(self, **signals):
        scores = {}
        for k, v in signals.items():
            if v is None or k not in self._mean:
                continue
            scores[k] = (v - self._mean[k]) / self._std[k]
        return scores

    def sample_count(self, key):
        return len(self._samples.get(key, []))

    def reset(self):
        self._samples.clear()
        self._mean.clear()
        self._std.clear()
        self.calibrated = False
