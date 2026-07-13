# Análise de Expressão Facial e Sinais Complementares (uso individual e local)

Protótipo que usa a webcam (e opcionalmente o microfone) para acompanhar,
em tempo real, sua emoção dominante (feliz, triste, raiva, nojo, medo,
surpresa, neutro) e um conjunto de sinais complementares — piscadas,
desvio de olhar, frequência cardíaca aproximada, postura e voz —
comparados com o **seu próprio baseline neutro**. Tudo roda localmente —
nenhuma imagem, áudio ou dado sai da sua máquina.

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
  mentira. O mesmo problema se aplica a qualquer sinal capturado aqui,
  voz incluída (hesitação e pausas indicam esforço cognitivo/nervosismo,
  não mentira especificamente).
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
- Todos os sinais complementares (BPM via rPPG, piscadas, gaze, postura,
  voz) são **aproximações grosseiras**, sensíveis a movimento, iluminação,
  qualidade da webcam/microfone e ruído ambiente. Nenhum deles substitui
  um exame clínico.
- O BPM tem uma checagem básica de qualidade (luz e movimento da testa) e
  fica oculto (`--`) quando a leitura não é confiável, em vez de mostrar
  um número que parece preciso mas não é.
- Classificação de emoção por IA tem taxa de erro relevante e viés
  conhecido (desempenho pior para alguns tons de pele e expressões
  culturais). Trate toda saída como sinal aproximado, não verdade
  absoluta.
- Se o objetivo for **clima organizacional de equipe**, a alternativa
  seria uma pesquisa de clima anônima e agregada — não câmeras/microfones
  individuais. Isso cria risco legal (LGPD, direito trabalhista) sem
  ganho real de precisão.

## Sinais capturados

| Sinal | Como | Limite |
|---|---|---|
| Emoção dominante | DeepFace (CNN pré-treinada) | macro-expressão, não microexpressão |
| Piscadas/min | MediaPipe Face Mesh, Eye Aspect Ratio (EAR) | sensível a óculos/ângulo da câmera |
| Desvio de olhar (gaze) | posição da íris vs. canto do olho (MediaPipe) | proxy grosseiro, não *eye tracking* calibrado |
| BPM aproximado (rPPG) | variação de cor da pele na testa + filtro passa-banda + FFT | cai muito com movimento/iluminação; some da tela se não confiável |
| Postura (ombros/tronco) | MediaPipe Pose | sinal grosseiro, útil só combinado com o resto |
| Volume/pausas/pitch da voz | RMS, detecção de silêncio e autocorrelação (microfone) | pitch é aproximado, não é um afinador profissional |
| Score de incongruência/estresse | z-score dos sinais acima contra o **seu** baseline neutro, suavizado (EMA) | só existe após calibrar (`b`); não indica mentira |

Todos os sinais numéricos passam por uma média móvel exponencial (EMA)
antes de aparecer na tela/log, pra reduzir tremulação sem esconder
tendência real.

## Instalação

```
pip install -r requirements.txt
```

`mediapipe`, `scipy` e `sounddevice` habilitam os sinais complementares
(piscadas, gaze, BPM, postura, voz). Se a instalação de algum deles
falhar no seu ambiente, o `main.py` continua funcionando com o que
estiver disponível — cada sinal é iniciado independentemente e avisa no
terminal qual ficou desativado, em vez de desligar tudo de uma vez.

Na primeira execução, o DeepFace baixa automaticamente os pesos do modelo
de emoção (uma única vez, fica em cache local). O microfone só é aberto
se o `sounddevice` conseguir inicializar um dispositivo de entrada — se
seu sistema pedir permissão de microfone, é esperado.

## Uso

```
python main.py
```

- `q` — sair (mostra um resumo da sessão no terminal: duração, % de
  tempo em cada emoção e % de tempo em cada nível de estresse)
- `l` — liga/desliga o log da sessão (grava em `session_log.csv`)
- `c` — **arquiva** o log atual em `session_archive/session_<data>.csv` e
  começa um novo (não apaga mais os dados, pra dar pra comparar sessões
  depois com `trends.py`)
- `b` — (re)inicia a calibração do baseline: fique ~12s parado, com
  expressão neutra, olhando pra câmera. Sem isso, o score de
  incongruência não aparece (mostra "sem baseline").

Depois de uma sessão com log ativado, gere um relatório pessoal:

```
python report.py
```

Isso gera `session_report.png` com a evolução das emoções, dos sinais
complementares (físicos e de voz) e do score de incongruência ao longo
**dessa sessão**.

Pra comparar **várias sessões ao longo do tempo** (ex.: seu nível médio
de estresse essa semana vs. semana passada), use:

```
python trends.py
```

Isso junta o `session_log.csv` atual com tudo que estiver em
`session_archive/` (arquivado com `c`), agrupa por dia e gera
`trend_report.png` + um resumo no terminal.

`session_log.csv`, `session_archive/` e os `.png` de relatório **não são
versionados** (estão no `.gitignore`) porque são dados pessoais — mantenha
assim, especialmente se este repositório for público.

## Build (.exe standalone)

Empacota só o `main.py` (a ferramenta de análise ao vivo) num `.exe` que
roda sem precisar instalar Python/dependências na máquina de destino.
`report.py` e `trends.py` continuam sendo scripts Python separados.

```
pip install -r requirements.txt -r requirements-build.txt
pyinstaller main.spec
```

O resultado fica em `dist/AnaliseExpressaoFacial/` — **precisa distribuir
a pasta inteira**, não só o `.exe` (é build "onedir", não "onefile", de
propósito: onefile re-extrai o TensorFlow inteiro pra uma pasta temporária
toda vez que abre, o que deixa o início bem lento).

Pontos de atenção, sem prometer que vai funcionar de primeira:

- **Tamanho**: espere algo entre 500 MB e 1,5 GB — TensorFlow/DeepFace são
  pesados, não tem muito o que fazer aqui.
- **Internet no primeiro uso**: o `.exe` builda sem os pesos do modelo de
  emoção do DeepFace (eles não fazem parte do pacote pip, são baixados em
  tempo de execução). Na máquina de destino, a primeira execução ainda
  precisa de internet pra baixar isso uma vez; depois fica em cache local
  (`~/.deepface/weights`).
- **`ModuleNotFoundError` no primeiro build**: `deepface`, `mediapipe` e
  `tensorflow` fazem import dinâmico em alguns pontos, que o analisador
  estático do PyInstaller pode não enxergar. O `main.spec` já inclui os
  hidden-imports mais prováveis (`collect_submodules('deepface')`,
  `mediapipe.python.solutions`, etc.), mas se faltar algo, o erro aponta
  o nome exato do módulo — adiciona ele na lista `hiddenimports` do
  `main.spec` e builda de novo. É normal precisar de 1-2 rodadas disso.
- `session_log.csv`/`session_archive/` são criados ao lado do `.exe`
  (não dentro da pasta de instalação do Python) quando rodando como
  build — o `main.py` detecta isso automaticamente.

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
- Trocar a estimativa de pitch por algo mais preciso (ex.: `pyin`/`crepe`)
  se a autocorrelação simples atual não for suficiente.
