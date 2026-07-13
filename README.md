# Análise de Expressão Facial (uso individual e local)

Protótipo que usa a webcam para detectar sua emoção dominante (feliz, triste,
raiva, nojo, medo, surpresa, neutro) em tempo real, usando um modelo
pré-treinado (DeepFace/OpenCV). Tudo roda localmente — nenhuma imagem sai da
sua máquina.

## Escopo e limites (leia antes de usar)

- Isto é uma ferramenta de **autoconhecimento pessoal**, não de vigilância ou
  avaliação de terceiros. Não use para gravar, pontuar ou ranquear colegas
  sem o consentimento explícito e informado de cada pessoa — isso pode violar
  a LGPD (dado biométrico é dado sensível) e normas trabalhistas.
- O modelo detecta **macro-expressões** (emoções que aparecem por ~1s ou
  mais). Micro-expressões reais (1/25 a 1/5 de segundo) exigem câmera de alto
  FPS (100+) e datasets especializados (CASME II, SAMM) — não é o que este
  protótipo faz.
- Classificação de emoção por IA tem taxa de erro relevante e viés conhecido
  (desempenho pior para alguns tons de pele e expressões culturais). Trate a
  saída como sinal aproximado, não verdade absoluta.

## Instalação

```
pip install -r requirements.txt
```

Na primeira execução, o DeepFace baixa automaticamente os pesos do modelo de
emoção (uma única vez, fica em cache local).

## Uso

```
python main.py
```

- `q` — sair
- `l` — liga/desliga o log da sessão (grava em `session_log.csv`)
- `c` — limpa o log

Depois de uma sessão com log ativado, gere um relatório pessoal:

```
python report.py
```

Isso gera `session_report.png` com a evolução das suas emoções ao longo do
tempo e a frequência de cada emoção dominante.

## Próximos passos possíveis

- Trocar o classificador de emoção por um modelo mais robusto (ex.: treinar
  em AffectNet) se a precisão do DeepFace não for suficiente.
- Adicionar MediaPipe Pose para incluir linguagem corporal (postura, ombros)
  como sinal complementar.
- Se o objetivo for clima organizacional de equipe, a alternativa mais segura
  é uma pesquisa de clima anônima e agregada, não câmeras individuais.
