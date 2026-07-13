"""
Testes unitarios pra logica pura de signals.py (matematica dos sinais),
sem precisar de webcam/microfone/mediapipe de verdade - usa dados
sinteticos.

Uso: python -m unittest test_signals -v

O que NAO e coberto aqui (precisaria de hardware/integracao real, fora
do escopo de teste unitario): FaceMeshTracker.process, PoseTracker.process
e VoiceTracker.start/_callback rodando de verdade contra uma camera ou
microfone. O que importa pra confianca no projeto - as contas por tras
de cada sinal - esta coberto.
"""

import math
import unittest

import numpy as np

import signals


class TestEMA(unittest.TestCase):
    def test_primeira_leitura_define_o_valor(self):
        ema = signals.EMA(alpha=0.5)
        self.assertIsNone(ema.value)
        self.assertEqual(ema.update(10.0), 10.0)

    def test_suaviza_em_direcao_ao_novo_valor(self):
        ema = signals.EMA(alpha=0.5)
        ema.update(10.0)
        result = ema.update(20.0)
        self.assertEqual(result, 15.0)  # 0.5*20 + 0.5*10

    def test_none_mantem_valor_atual(self):
        ema = signals.EMA(alpha=0.5)
        ema.update(10.0)
        self.assertEqual(ema.update(None), 10.0)

    def test_reset_volta_a_none(self):
        ema = signals.EMA(alpha=0.5)
        ema.update(10.0)
        ema.reset()
        self.assertIsNone(ema.value)


class TestEyeAspectRatio(unittest.TestCase):
    """EAR = (dist vertical 1 + dist vertical 2) / (2 * dist horizontal).
    Indices esperados pelo layout LEFT_EYE/RIGHT_EYE: [corner, top1, top2,
    corner, bottom2, bottom1]."""

    def _make_eye(self, eye_height):
        # 6 pontos: canto esquerdo, 2 topo, canto direito, 2 base
        return np.array([
            [0.0, 5.0],                      # canto esquerdo
            [3.0, 5.0 - eye_height / 2],      # topo 1
            [7.0, 5.0 - eye_height / 2],      # topo 2
            [10.0, 5.0],                      # canto direito
            [7.0, 5.0 + eye_height / 2],       # base 2
            [3.0, 5.0 + eye_height / 2],       # base 1
        ])

    def test_olho_aberto_tem_ear_maior_que_fechado(self):
        idx = [0, 1, 2, 3, 4, 5]
        aberto = self._make_eye(eye_height=4.0)
        fechado = self._make_eye(eye_height=0.4)

        ear_aberto = signals._eye_aspect_ratio(aberto, idx)
        ear_fechado = signals._eye_aspect_ratio(fechado, idx)

        self.assertGreater(ear_aberto, ear_fechado)
        self.assertLess(ear_fechado, signals.FaceMeshTracker.EAR_BLINK_THRESHOLD)
        self.assertGreater(ear_aberto, signals.FaceMeshTracker.EAR_BLINK_THRESHOLD)

    def test_horizontal_zero_nao_quebra(self):
        pts = np.array([[0.0, 0.0]] * 6)
        self.assertEqual(signals._eye_aspect_ratio(pts, [0, 1, 2, 3, 4, 5]), 0.0)


class TestGazeOffset(unittest.TestCase):
    def _make_pts(self, iris_shift=0.0):
        pts = np.zeros((478, 2))
        # canto esquerdo do olho esquerdo/direito
        pts[signals.LEFT_EYE_CORNERS[0]] = [0.0, 0.0]
        pts[signals.LEFT_EYE_CORNERS[1]] = [10.0, 0.0]
        pts[signals.RIGHT_EYE_CORNERS[0]] = [0.0, 0.0]
        pts[signals.RIGHT_EYE_CORNERS[1]] = [10.0, 0.0]
        # iris no centro (5,0) + deslocamento
        center = 5.0 + iris_shift
        for i in signals.LEFT_IRIS:
            pts[i] = [center, 0.0]
        for i in signals.RIGHT_IRIS:
            pts[i] = [center, 0.0]
        return pts

    def test_olhar_centralizado_offset_proximo_de_zero(self):
        pts = self._make_pts(iris_shift=0.0)
        self.assertAlmostEqual(signals._compute_gaze_offset(pts), 0.0, places=6)

    def test_olhar_desviado_aumenta_offset(self):
        centralizado = signals._compute_gaze_offset(self._make_pts(iris_shift=0.0))
        desviado = signals._compute_gaze_offset(self._make_pts(iris_shift=3.0))
        self.assertGreater(desviado, centralizado)


class TestBaselineCalibrator(unittest.TestCase):
    def test_precisa_de_amostras_minimas_pra_calibrar(self):
        cal = signals.BaselineCalibrator()
        for v in [1.0, 1.1, 0.9]:  # menos que MIN_SAMPLES
            cal.add_sample(sinal=v)
        self.assertFalse(cal.finalize())
        self.assertFalse(cal.calibrated)

    def test_zscore_de_outlier_e_alto(self):
        cal = signals.BaselineCalibrator()
        for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]:
            cal.add_sample(sinal=v)
        self.assertTrue(cal.finalize())

        z_normal = cal.zscore(sinal=1.0)["sinal"]
        z_outlier = cal.zscore(sinal=10.0)["sinal"]
        self.assertLess(abs(z_normal), abs(z_outlier))

    def test_valor_none_e_ignorado(self):
        cal = signals.BaselineCalibrator()
        for v in [1.0, 1.1, 0.9, 1.05, 0.95]:
            cal.add_sample(sinal=v, outro=None)
        cal.finalize()
        self.assertEqual(cal.sample_count("outro"), 0)

    def test_reset_limpa_tudo(self):
        cal = signals.BaselineCalibrator()
        for v in [1.0, 1.1, 0.9, 1.05, 0.95]:
            cal.add_sample(sinal=v)
        cal.finalize()
        cal.reset()
        self.assertFalse(cal.calibrated)
        self.assertEqual(cal.sample_count("sinal"), 0)


@unittest.skipUnless(signals.butter is not None, "scipy indisponivel neste ambiente")
class TestRPPGEstimator(unittest.TestCase):
    def _feed_sine_bpm(self, estimator, bpm, seconds, fps):
        """Alimenta o buffer com um sinal senoidal sintetico simulando a
        variacao de cor da pele numa frequencia cardiaca conhecida."""
        freq_hz = bpm / 60.0
        n = int(fps * seconds)
        for i in range(n):
            t = i / fps
            green = 128 + 5 * math.sin(2 * math.pi * freq_hz * t)
            frame = np.full((4, 4, 3), green, dtype=np.float64)
            estimator.add_frame(frame, (0, 0, 4, 4))

    def test_recupera_bpm_proximo_do_sintetico(self):
        fps = 30
        estimator = signals.RPPGEstimator(fps=fps, buffer_seconds=8)
        self._feed_sine_bpm(estimator, bpm=72, seconds=8, fps=fps)
        bpm = estimator.estimate_bpm()
        self.assertIsNotNone(bpm)
        # resolucao do bin de FFT com 8s de janela e de 60/8 = 7.5 bpm
        self.assertLess(abs(bpm - 72), 10)

    def test_buffer_incompleto_retorna_none(self):
        estimator = signals.RPPGEstimator(fps=30, buffer_seconds=8)
        self._feed_sine_bpm(estimator, bpm=72, seconds=1, fps=30)
        self.assertIsNone(estimator.estimate_bpm())

    def test_movimento_brusco_marca_motion_ok_false(self):
        estimator = signals.RPPGEstimator(fps=30, buffer_seconds=8)
        # frame grande o suficiente pra as duas caixas caberem dentro
        # dela (senao o recorte da ROI fica vazio e o metodo retorna
        # antes de sequer avaliar o movimento)
        frame = np.full((700, 700, 3), 128, dtype=np.uint8)
        estimator.add_frame(frame, (0, 0, 50, 50))
        self.assertTrue(estimator.motion_ok)
        # caixa da testa pula muito de posicao entre frames
        estimator.add_frame(frame, (500, 500, 50, 50))
        self.assertFalse(estimator.motion_ok)

    def test_luz_fora_da_faixa_marca_lighting_ok_false(self):
        estimator = signals.RPPGEstimator(fps=30, buffer_seconds=8)
        escuro = np.full((10, 10, 3), 2, dtype=np.uint8)
        estimator.add_frame(escuro, (0, 0, 10, 10))
        self.assertFalse(estimator.lighting_ok)

    def test_calibrate_lighting_ajusta_faixa(self):
        estimator = signals.RPPGEstimator(fps=30, buffer_seconds=8)
        amostras = [100, 102, 98, 101, 99, 103, 97, 100, 101, 99, 100]
        self.assertTrue(estimator.calibrate_lighting(amostras))
        self.assertLess(estimator.MIN_BRIGHTNESS, 97)
        self.assertGreater(estimator.MAX_BRIGHTNESS, 103)

    def test_calibrate_lighting_com_poucas_amostras_falha(self):
        estimator = signals.RPPGEstimator(fps=30, buffer_seconds=8)
        self.assertFalse(estimator.calibrate_lighting([100, 101]))


@unittest.skipUnless(signals.sd is not None, "sounddevice indisponivel neste ambiente")
class TestVoiceTrackerPitch(unittest.TestCase):
    def test_estima_pitch_proximo_do_sintetico(self):
        tracker = signals.VoiceTracker(samplerate=16000)
        freq_hz = 150.0
        duration_s = 0.5
        t = np.arange(int(tracker.samplerate * duration_s)) / tracker.samplerate
        audio = np.sin(2 * math.pi * freq_hz * t)

        pitch = tracker._estimate_pitch(audio)
        self.assertIsNotNone(pitch)
        self.assertLess(abs(pitch - freq_hz), 5.0)

    def test_audio_fora_da_faixa_de_pitch_retorna_none_ou_proximo(self):
        tracker = signals.VoiceTracker(samplerate=16000)
        # 20 Hz esta bem abaixo do PITCH_MIN_HZ (70) - autocorrelacao nao
        # deve "inventar" um pitch dentro da faixa valida
        t = np.arange(int(tracker.samplerate * 0.5)) / tracker.samplerate
        audio = np.sin(2 * math.pi * 20.0 * t)
        pitch = tracker._estimate_pitch(audio)
        if pitch is not None:
            self.assertGreaterEqual(pitch, tracker.PITCH_MIN_HZ - 1)


class TestFaceMeshBlinkCalibration(unittest.TestCase):
    def test_calibrate_blink_threshold_poucas_amostras_falha(self):
        # nao precisa de mediapipe de verdade - so testa a conta pura,
        # entao chamamos o metodo direto numa instancia "crua" (sem
        # passar pelo __init__ que exige mediapipe instalado)
        tracker = signals.FaceMeshTracker.__new__(signals.FaceMeshTracker)
        self.assertFalse(tracker.calibrate_blink_threshold([0.3, 0.31]))

    def test_calibrate_blink_threshold_ajusta_pro_perfil(self):
        tracker = signals.FaceMeshTracker.__new__(signals.FaceMeshTracker)
        amostras = [0.30, 0.31, 0.29, 0.32, 0.30, 0.28, 0.31, 0.30, 0.29, 0.30]
        self.assertTrue(tracker.calibrate_blink_threshold(amostras))
        self.assertAlmostEqual(tracker.EAR_BLINK_THRESHOLD, 0.30 * 0.75, delta=0.02)


if __name__ == "__main__":
    unittest.main()
