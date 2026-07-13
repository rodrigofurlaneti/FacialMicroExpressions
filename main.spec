# -*- mode: python ; coding: utf-8 -*-
#
# Build standalone (onedir) do main.py com PyInstaller.
#
# Uso (no venv, com requirements.txt + requirements-build.txt instalados):
#   pyinstaller main.spec
#
# Resultado: dist/AnaliseExpressaoFacial/AnaliseExpressaoFacial.exe
#
# Por que onedir e nao onefile: onefile extrai tensorflow/mediapipe
# inteiros pra uma pasta temporaria TODA VEZ que o .exe abre, o que deixa
# o start bem lento (30s a 1min). Onedir descompacta uma vez (no build) e
# depois abre quase instantaneo. Pra distribuir, zipa a pasta
# dist/AnaliseExpressaoFacial inteira, nao so o .exe.
#
# So empacota main.py (a ferramenta de analise ao vivo). report.py e
# trends.py continuam sendo scripts Python separados - se quiser
# executaveis deles tambem, cria um .spec por script.
#
# report.py/trends.py que usam pandas/matplotlib, o main.py atual nao
# importa nenhum dos dois diretamente - se isso mudar, tira eles daqui
# de qualquer exclude que vier a existir.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
datas += collect_data_files("mediapipe")    # modelos .tflite embutidos no pacote
datas += collect_data_files("sounddevice")  # DLL do PortAudio

# deepface e mediapipe fazem parte dos imports por nome (string) em vez
# de import estatico em alguns pontos - o analisador do PyInstaller nao
# enxerga isso sozinho, entao declaramos os pacotes inteiros como hidden
# imports pra garantir que tudo que eles podem carregar em runtime vai
# junto no build.
hiddenimports = []
hiddenimports += collect_submodules("deepface")
hiddenimports += collect_submodules("mediapipe.python.solutions")
hiddenimports += ["scipy.signal", "sounddevice", "cv2"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AnaliseExpressaoFacial",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # mantem o terminal - o script imprime avisos/resumo nele
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AnaliseExpressaoFacial",
)
