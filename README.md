# Análise de Expressão Facial e Sinais Complementares (uso individual e local)

Protótipo que usa a webcam para acompanhar, em tempo real, sua emoção
dominante (feliz, triste, raiva, nojo, medo, surpresa, neutro) e um
conjunto de sinais complementares — piscadas, desvio de olhar, frequência
cardíaca aproximada e postura — comparados com o **seu próprio baseline
neutro**. Tudo roda localmente — nenhuma imagem sai da sua máquina.

## Por que isto não é (e não vira) um detector de mentira

Esse era o objetivo inicial do projeto, e vale deixar registrado por que
mudamos de direção:

- **Não existe hoje um método validado cientificamente** que detecte
  mentira de forma confiável a partir de sinais faciais/fisiológicos
  isolados — nem microexpressões, nem polígrafo, nem IA. A metanálise
  mais citada da área (Bond & DePaulo, ~200 estudos) mostra humanos e a
  maioria dos métodos automatizados acertando por volta de **54%**,
  pouco acima de jogar uma moeda (50%).
- As claims originais de Paul Ekman sobre microexpressões como "sinal de
  mentira" foram bastante contestadas por pesquisa posterior. O que tem
  base sólida é que microexpressões indicam **emoção reprimida ou
  contida** — não necessariamente engano.
- É por isso que o **polígrafo é inadmissível como prova** na maioria
  dos tribunais: mede excitação fisiológica (ansiedade, estresse), não
  mentira. O mesmo problema se aplica a qualquer sinal capturado aqui.
- **Risco prático de usar isso em outra pessoa**: dado biométrico é dado
  sensível pela LGPD, e um falso positivo (rotular alguém de "estressado"
  ou, pior, de "mentindo") pode prejudicar essa pessoa injustamente sem
  respaldo científico nenhum por trás.

Por isso o projeto foi reformulado: em vez de "está mentindo ou não", ele
mede **incongruência emocional / picos de estresse em relação ao seu
próprio normal** — um sinal para você refletir, não um veredito
automático.

## Escopo e limites (leia antes de usar)

- Isto é uma ferramenta de **autoconhecimento pessoal**, não de
  vigilância ou avaliação de terceiros. Não use para gravar, pontuar ou
  ranquear colegas sem o consentimento explícito e informado de cada
  pessoa.
- O modelo de emoção detecta **macro-expressões** (≈1s ou mais).
  Microexpressões reais (1/25 a 1/5 de segundo) exigem câmera de alto FPS
  (100+) e datasets especializados (CASME II, SAMM) — não é o que este
  protótipo faz, e mesmo com isso o link para "mentira" continuaria fraco
  (ver seção acima).
- Todos os sinais complementares (BPM via rPPG, piscadas, gaze, postura)
  são **aproximações grosseiras**, sensíveis a movimento, iluminação e
  qualidade da webcam. Nenhum deles substitui um exame clínico.
- Classificação de emoção por IA tem taxa de erro relevante e viés
  conhecido (desempenho pior para alguns tons de pele e expressões
  culturais). Trate toda saída como sinal aproximado, não verdade
  absoluta.
- Se o objetivo for **clima organizacional de equipe**, a alternativa
  seria uma pesquisa de clima anônima e agregada — não câmeras
  individuais. Câmeras individuais nesse contexto criam risco legal
  (LGPD, direito trabalhista) sem ganho real de precisão.

## Sinais capturados

| Sinal | Como | Limite |
|---|---|---|
| Emoção dominante | DeepFace (CNN pré-treinada) | macro-expressão, não microexpressão |
| Piscadas/min | MediaPipe Face Mesh, Eye Aspect Ratio (EAR) | sensível a óculos/ângulo da câmera |
| Desvio de olhar (gaze) | posição da íris vs. canto do olho (MediaPipe) | proxy grosseiro, não *eye tracking* calibrado |
| BPM aproximado (rPPG) | variação de cor da pele na testa + filtro passa-banda + FFT | cai muito com movimento/iluminação; não é clínico |
| Postura (ombros/tronco) | MediaPipe Pose | sinal grosseiro, útil só combinado com o resto |
| Score de incongruência/estresse | z-score dos sinais acima contra o **seu** baseline neutro | só existe após calibrar (`b`); não indica mentira |

## Instalação

```
pip install -r requirements.txt
```

`mediapipe` e `scipy` habilitam os sinais complementares (piscadas, gaze,
BPM, postura). Se a instalação deles falhar no seu ambiente (ex.: versão
do Python incompatível), o `main.py` continua funcionando só com a
detecção de emoção — ele avisa no terminal e desativa o painel extra
automaticamente.

Na primeira execução, o DeepFace baixa automaticamente os pesos do modelo
de emoção (uma única vez, fica em cache local).

## Uso

```
python main.py
```

- `q` — sair
- `l` — liga/desliga o log da sessão (grava em `session_log.csv`)
- `c` — limpa o log
- `b` — (re)inicia a calibração do baseline: fique ~12s parado, com
  expressão neutra, olhando pra câmera. Sem isso, o score de
  incongruência não aparece (mostra "sem baseline").

Depois de uma sessão com log ativado, gere um relatório pessoal:

```
python report.py
```

Isso gera `session_report.png` com a evolução das emoções, dos sinais
complementares e do score de incongruência ao longo da sessão.

## Próximos passos possíveis

- **Trocar/complementar o classificador de emoção** (ex.: fine-tuning em
  AffectNet) se a precisão do DeepFace não for suficiente. Isso não foi
  feito neste projeto porque exige: (1) licenciar/baixar o dataset
  (AffectNet requer pedido de acesso acadêmico), (2) GPU para treino,
  (3) uma pipeline de avaliação separada pra não sobreajustar a poucas
  pessoas/condições de luz. Vale a pena só se o DeepFace atual estiver
  errando muito no seu caso de uso real — meça antes de trocar.
- Refinar o `gaze_offset` para uma calibração de tela (mapear onde
  exatamente na tela a pessoa está olhando), hoje é só magnitude de
  desvio.
- Ajustar os índices de landmark do MediaPipe (`signals.py`) se a versão
  instalada mudar a numeração dos pontos de íris.
