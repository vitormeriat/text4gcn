# Plano de Melhorias — Text4GCN

> Atualizado em 2026-08-05 após: (1) validação de engenharia do pacote; (2) leitura do
> paper original do pacote (Pereira & de Castro, DCAI 2022) e da apresentação em
> `papers/`; (3) cruzamento com Yao et al. (AAAI 2019); (4) posicionamento frente ao
> estado da arte 2021–2025 (BertGCN, InducT-GCN, surveys GCN-for-text, críticas a
> progresso em text classification).
>
> Cada seção abaixo é **independente** — pode ser resolvida isoladamente dentro da
> mesma fase. Marque o checkbox após validar com o comando indicado.

---

## Changelog

Registo de execução deste plano (mais recente no topo).

### 2026-08-05 — Plano fechado (P1–P5 + extras)

**Status:** itens de engenharia e rigor do plano **implementados ou explicitamente adiados**.

| Item | Estado | Notas |
|---|---|---|
| **#6** di/optimization/services | ✅ Removidos | ~6k LOC órfãos + exemplos |
| **#5** trainer composto | ✅ Removido | 7 ficheiros em `models/` |
| **#7** testes obsoletos | ✅ Removidos | DI/opt/refactor tests |
| **#2** AsyncProcessor | ✅ Some com #6 | |
| **#11** `epoches` | ✅ Some com #5 | |
| **#9** artefatos Jekyll | ✅ | `.gitignore` + `git rm --cached` (se versionados) |
| **#8** lint crítico | ✅ | flake8 select E9/F82/F821/E722/F401/F841 = **0** no pacote |
| **#13** CONTRIBUTING | ✅ | `docs/contributing.md` aponta para raiz |
| **#12** CoreNLP | ✅ | Extra `pip install text4gcn[syntactic]`; import lazy |
| **#15** GAT | ✅ (P0) | Removido do enum; sem implementação fantasma |
| **#16** CI | ✅ | flake8 hard + pytest core + reproduce smoke; mypy soft (#10) |
| **#10** mypy | ⚠️ Baseline reduzida | Strict flags relaxados; CI soft até tipagem completa |
| **#14** BertGCN híbrido | ⏸️ Roadmap | Não é claim DCAI; só após suite Tab. 3 estável |
| **#20** inductive/subgraphs | ⏸️ Pesquisa | Futuro do próprio paper |

**Removido do tree:** `text4gcn/{di,optimization,services}/`, trainer composto, testes/exemplos associados.

**Validação:**
```bash
flake8 text4gcn --max-line-length=100 --select=E9,F63,F7,F82,F821,E722,F401,F841 --count  # 0
pytest tests/ -q -o addopts=
python scripts/reproduce_dcai2022.py --reference-only
```

### 2026-08-05 — Fase A1 concluída (âncora no paper)

**Status:** feitos #17, #18, #19.

| Item | Mudança | Arquivos |
|---|---|---|
| **#18** Stats de arestas | `GraphBuildStats` + `compute_adjacency_stats` / `load_adjacency_stats`; Frequency/Word2Vec retornam stats e logam nnz/tamanho | `graph/stats.py`, `builder/frequency_adj.py`, `builder/word2vec_adj.py`, `builder/pipeline.py`, `__init__.py` |
| **#17** Reprodução Tab. 2/3 | Script com tabelas de referência do DCAI 2022, comparação de ordem de grandeza de arestas, treino opcional | `scripts/reproduce_dcai2022.py`, `tests/test_graph_stats_a1.py` |
| **#19** Docs honestos | README: nicho RWF/memória, limitação **transdutiva**, protocolo §3.2, citação DCAI 2022 | `README.md` |

**Validação:**
```bash
python scripts/reproduce_dcai2022.py --reference-only
pytest tests/test_graph_stats_a1.py -q -o addopts=
```

**Próxima fase sugerida:** P1 (#6 → #5 → #7 enxugar código órfão).

### 2026-08-05 — Fase P0 concluída (bugs do path publicado)

**Status:** feitos #1, #3, #4. Item #2 (AsyncProcessor) **adiado de propósito** — código órfão; remover com #6.

| Item | Mudança | Arquivos |
|---|---|---|
| **#1** `dia_matrix` | Features/adj convertidos para `csr_matrix` antes de fancy indexing | `graph/refinement.py`, `tests/test_graph_refinement.py` |
| **#3** Validação no `__init__` | `GNN` e `TextPipeline` falham cedo se `dataset_path` não existe ou `dataset_name` vazio; layer inválida rejeitada no init | `models/gnn.py`, `preprocess/pipeline.py` |
| **#4** `Layer` honesto | Removido `Layer.GAT` do enum público; docs e `config_manager` alinhados a só GCN | `models/layers.py`, `models/config_manager.py`, `docs/api/gnn.md` |

**Validação:**
```bash
pytest tests/test_graph_refinement.py tests/test_models.py::TestGNNErrorHandling \
  tests/test_integration.py::TestErrorHandling tests/test_academic_rigor_a0.py -q -o addopts=
```

**Próxima fase sugerida:** A1 (#17–#19 reprodução Tab. 2/3 + docs) **ou** P1 (#6→#5 enxugar órfãos).

### 2026-08-05 — Fase A0 concluída (rigor acadêmico)

**Status:** feitos 0a, 0b, 0c, 0d.

| Item | Mudança | Arquivos |
|---|---|---|
| **0a** Seed | `GNN._build_args` usa `self.seed`; `DeviceManager.set_seed` cobre `random`/`numpy`/`torch`(+CUDA); removido `th.seed()` que re-sorteava o RNG; preprocess shuffle seedado | `models/gnn.py`, `models/train_eval.py`, `utils/device.py`, `preprocess/pipeline.py` |
| **0a** Dropout | `TextGCNTrainer.fit` passa `dropout_rate=args.dropout` ao `GCN`; `model.train()` a cada epoch (antes `_evaluate` deixava o modelo em `eval`) | `models/train_eval.py` |
| **0b** Early-stop | Guarda `best_state` pelo menor val loss; restaura antes do teste e do checkpoint | `models/train_eval.py` |
| **0c** `val_ratio` | Docstring + `DeprecationWarning` se ≠ 0.1; split continua só em `TextPipeline.execute(validation_ratio=...)`; `seed` adicionado ao pipeline | `models/gnn.py`, `preprocess/pipeline.py` |
| **0d** RWF pesos | Peso Max–Min (Eq. 6 DCAI); PMI só como gate; removido z-score; `relation_pair_statitcs` com min correto; builders Dependency/LIWC delegam ao helper partilhado | `modules/adjacency.py`, `modules/word_processor.py`, `builder/dependency_parsing_adj.py`, `builder/linguistic_inquiry_adj.py` |

**Testes:** `tests/test_academic_rigor_a0.py` (7 passed).

```bash
PYTHONPATH=.:.venv/lib/python3.11/site-packages python -m pytest tests/test_academic_rigor_a0.py -q -o addopts=
```

**Próxima fase sugerida:** P0 (#1 `dia_matrix`, #3 validação, #4 Layer) → depois A1 (#17–#19).

---

## Posicionamento científico (ler antes de priorizar)

### O que o paper do pacote reivindica (DCAI 2022)

**Referência:** Pereira, V. C. M. & de Castro, L. N. *Natural Language Processing Based
on a Text Graph Convolutional Network*. DCAI 2022. Fontes: `papers/DCAI_2022_TGCN_paper.pdf`,
`papers/DCAI_2022_TGCN_presentation.pdf`.

**Contribuição central — não é “bater SOTA de acurácia”.** O paper propõe o
**Related Word Filter (RWF)**: filtros sintáticos/semânticos/linguísticos que **reduzem
drasticamente o número de arestas** do grafo heterogêneo word–document do TGCN
(Yao et al.), mantendo acurácia próxima (e às vezes ligeiramente superior) ao TGCN
padrão, com menor memória e tempo de treino.

| Claim do paper | Evidência no paper | Implicação para o pacote |
|---|---|---|
| TGCN puro consome memória excessiva | Tab. 2: 20NG ≈ 24.7M arestas | O valor do pacote é **construção de grafo filtrado**, não só treinar GCN |
| RWF reduz arestas em ordem de magnitude | Tab. 2: SEMANTIC 20NG ≈ 2.3M (−91%) | Builders Semantic/Syntactic/Linguistic são o diferencial — devem ser corretos e mensuráveis |
| Acurácia ≈ TGCN (às vezes +) | Tab. 3: R8 SEMANTIC 0.969 vs TGCN 0.968 | Não compete com BERT/BertGCN; compete com TGCN **em eficiência** |
| Protocolo Yao | dropout 0.5, lr 0.02, nhid 200, early stop 10, val 10% | Pacote deve **reproduzir** esse protocolo de forma fiel |
| Módulos RWF | Dependency (CoreNLP), LIWC, Word2Vec cosine > 0.95 | Código precisa bater com Eq. (6)–(8) do paper (peso e gate) |

**Protocolo experimental do paper (§3.2):** 2 camadas GCN, dropout **0.5**, embedding 200,
Adam lr **0.02**, max 200 epochs, early stopping 10, 10% do train como val. Baselines:
TGCN, CNN, LSTM. Datasets: 20NG, MR, R8, R52, CORA, PubMed, Citeseer.

**Futuros trabalhos citados pelo próprio paper:** grafos mais representativos com embeddings;
processamento por **subgrafos** para corpora grandes. Isso ainda é aberto e alinhado a
direções atuais (sampling / inductive).

### Yao et al. (AAAI 2019) — base metodológica

TextGCN: grafo corpus-level (docs + words), arestas word–word via **PMI > 0**, doc–word via
**TF-IDF**, features identidade, GCN 2 camadas featureless, aprendizado **transdutivo**.
Continua sendo a referência de baseline em surveys (ex.: Rizvi et al. 2024, *Text Classification
using GCNs: A Comprehensive Survey*).

### Estado da arte atual (2021–2025) — o pacote ainda é relevante?

**Veredito curto:** sim, **como toolkit de pesquisa em construção de text graphs e TGCN
filtrado**; não, **como classificador SOTA genérico**. O nicho do DCAI 2022 permanece
válido; a relevância depende de rigor experimental e honestidade científica da API.

| Frente | Onde está o campo (2024–25) | Onde está o Text4GCN | Lacuna |
|---|---|---|---|
| Acurácia benchmark (R8/R52/20NG) | BertGCN / RoBERTaGCN ~98.1 / 96.6 / 89.5; PLMs fine-tuned dominam textos curtos | TGCN-style ~97 / 93 / 86 (paper Yao); RWF ≈ isso | **Não é o claim do paper** — não priorizar BertGCN antes de fidelidade RWF |
| Híbridos PLM+GNN | Lin et al. 2021 (BertGCN); VGCN-BERT; surveys 2024 enfatizam híbridos | `bert_adj` = similaridade para **arestas**, não BertGCN | Naming enganoso; híbrido real = feature opcional (fase P4) |
| Indutivo vs transdutivo | InducT-GCN, TextING, HyperGAT; crítica a modelos que exigem test no grafo | Strictamente **transdutivo** (como Yao e como o paper DCAI) | Limitação conhecida; documentar; inductive = pesquisa futura (paper já cita subgraphs) |
| Representação de grafo | Bugueño & de Melo 2023; surveys: tipo de grafo importa tanto quanto a GNN | Frequency (PMI) + RWF builders | **É o diferencial**; precisa ser cientificamente correto |
| Memória / escala | Sampling, subgraphs, sparsification | RWF + `graph/refinement` | Alinhado ao paper; refinement quebrado (#1) prejudica o claim |
| Crítica metodológica | Galke & Scherp (*Are We Really Making Much Progress…*): muitos gains GNN são frágeis vs baselines fortes | Sem suite de reprodução Tab. 2/3; seed/dropout quebrados | Risco de **não poder defender o paper** com o código atual |
| Ecossistema eng. | PyG / DGL para camadas GNN; poucos libs focadas em **text-graph builders** | API `TextPipeline` + builders — nicho quase vazio | Manter foco em **builders + RWF**, não virar mini-PyG |

**Conclusão de relevância:** o pacote não está obsoleto. Transformers ganharam a guerra de
acurácia em textos curtos; grafos de texto continuam úteis para (a) regimes com poucos rótulos,
(b) documentos longos / coocorrência global, (c) estudo de **como** construir o grafo,
(d) eficiência de memória — exatamente o enquadramento DCAI 2022. O que ameaça a relevância
hoje não é o SOTA de 2025: é **código que não reproduz o protocolo do próprio paper** e
infraestrutura morta que dilui o foco científico.

### Critério para aceitar/rejeitar itens deste plano

1. **P0 acadêmico** — sem isso, números do paper não são defensáveis.
2. **Fidelidade RWF** — claim central do DCAI 2022.
3. **Higiene do path publicado** — o que `GNN`/`TextPipeline`/`Frequency`/builders realmente usam.
4. **Competitividade opcional** — BertGCN, GAT, inductive: só depois de 1–3.
5. **Não fazer:** “framework genérico de DI/cache/async” sem uso no pipeline científico.

---

## Sumário

| # | Item | Fase | Esforço | Risco de não fazer |
|---|---|---|---|---|
| [0a](#0a-reprodutibilidade-quebrada-seed-dropout-shuffle) | Reprodutibilidade: seed / dropout / shuffle | **A0** ✅ | pequeno | **Crítico** — invalida claims experimentais |
| [0b](#0b-early-stop-avalia-o-modelo-errado) | Early-stop avalia o modelo errado | **A0** ✅ | pequeno | Alto — métricas de teste enviesadas |
| [0c](#0c-gnnval_ratio-é-no-op) | `GNN(val_ratio=…)` é no-op | **A0** ✅ | pequeno | Médio — ablations silenciosas |
| [0d](#0d-fidelidade-rwf-peso-das-arestas-≠-paper) | Fidelidade RWF: peso ≠ Eq. (6) do paper | **A0** ✅ | médio | **Crítico** — claim central não reproduzível |
| [1](#1-bug-apply_refinement-quebra-com-dia_matrix) | Bug `apply_refinement` + `dia_matrix` | P0 ✅ | pequeno | Alto — feature do claim de memória |
| [2](#2-bug-ids-trocados-em-asyncprocessor) | IDs no `AsyncProcessor` | P0 ✅ (removido c/ #6) | pequeno | Baixo — preferir remover (#6) |
| [3](#3-gnn-e-textpipeline-não-validam-entrada-no-init) | Validação de entrada | P0 ✅ | pequeno | Médio — UX |
| [4](#4-layer-inconsistente--api-promete-gat) | `Layer` inconsistente / GAT fantasma | P0 ✅ | pequeno | Médio — API vs paper (só GCN no experimento) |
| [5](#5-965-linhas-de-trainer-composto-nunca-conectado) | Trainer composto órfão | P1 ✅ | médio | Alto — manutenção |
| [6](#6-5930-linhas-de-diloptimizationservices-não-usadas) | `di/` / `optimization/` / `services/` | P1 ✅ | médio | Alto — 56% do LOC, 0% do paper |
| [7](#7-testes-obsoletos-do-refactor-anterior) | Testes obsoletos | P1 ✅ | pequeno | Baixo, suja CI |
| [8](#8-lint-flake8) | Lint flake8 | P1 ✅ | pequeno | Baixo |
| [9](#9-artefatos-de-build-versionados-no-git) | Artefatos Jekyll no git | P1 ✅ | pequeno | Baixo |
| [10](#10-mypy) | Erros mypy | P2 ⚠️ soft | médio | Médio |
| [11](#11-typo-epoches) | Typo `epoches` | P2 ✅ (sumiu c/ #5) | trivial | Baixo (só órfão) |
| [12](#12-stanford-corenlp--fricção-vs-módulo-sintático-do-paper) | Stanford CoreNLP | P3 ✅ | grande | Alto p/ Syntactic do paper |
| [13](#13-contributingmd-duplicado) | CONTRIBUTING duplicado | P3 ✅ | trivial | Baixo |
| [17](#17-suite-de-reprodução-das-tabelas-23-do-paper) | Suite de reprodução Tab. 2/3 | **A1** ✅ | médio | Alto — sem ela o paper não se ancora no código |
| [18](#18-métricas-de-arestasmemória-expostas-na-api) | Reportar #arestas / memória na API | **A1** ✅ | pequeno | Médio — claim Tab. 2 invisível ao usuário |
| [19](#19-documentar-limitação-transdutiva-e-nicho) | Docs: transdutivo + nicho vs transformers | **A1** ✅ | pequeno | Médio — expectativa errada de SOTA |
| [14](#14-híbrido-bertgcn--opcional-não-é-o-claim-do-paper) | Híbrido BertGCN | P4 ⏸️ roadmap | grande | Baixo p/ paper; médio p/ marketing |
| [15](#15-implementar-gat-de-verdade-ou-remover-do-enum) | GAT real ou remover | P4 ✅ (removido) | grande/pequeno | Médio |
| [16](#16-ci-reprodutibilidade--lint) | CI: seed fixo + lint/mypy | P5 ✅ | pequeno | Alto a longo prazo |
| [20](#20-pesquisa-futura-alinhada-ao-paper-inductive--subgraphs) | Inductive / subgraphs (futuro paper) | P5+ ⏸️ pesquisa | grande | — pesquisa, não dívida |

---

## Contexto de engenharia (validação 2026-08-05)

**Validado:** `pytest`, `flake8`, `mypy --ignore-missing-imports`, grep de referências,
leitura do path vivo (`Frequency` → `adjacency.extract_pmi_*` → `train_eval.TextGCNTrainer`).

```
201 passed, 22 failed, 1 skipped
```

**Núcleo acadêmico vivo (~829 LOC):** `preprocess/pipeline.py`, `builder/frequency_adj.py`,
builders RWF, `modules/adjacency.py`, `models/{gnn,train_eval,model,utils}.py`.

**Código órfão (~56% do pacote):** `di/`, `optimization/`, `services/`, trainer composto —
**zero contribuição** às Tabelas 2/3 do paper.

---

## FASE A0 — Rigor acadêmico (antes de tudo)

### 0a. Reprodutibilidade quebrada: seed, dropout, shuffle ✅

**Arquivos:** `models/gnn.py` (`_build_args`), `models/train_eval.py` (`_train`, `fit`),
`preprocess/pipeline.py` (`shuffle_data`).

**Evidência:**
```python
# gnn.py:_build_args — ignora self.seed
seed=np.random.randint(1, 100_000),

# train_eval.py:_train — th.seed() RE-SORTEIA o RNG após manual_seed
th.manual_seed(self.args.seed)
print(f"... Torch Seed: {th.seed()}")   # th.seed() sem args = novo seed aleatório

# train_eval.py:fit — GCN sem dropout_rate → default 0.0
self.model = self.model(input_dim=..., output_dim=..., support=..., num_classes=...)
# paper DCAI §3.2 e Yao: dropout 0.5; GNN(dropout=0.5) é anunciado mas nunca aplicado

# preprocess: random.shuffle(...) sem seed
```

**Por que é A0:** o paper DCAI e Yao reportam acurácias com protocolo fixo. Sem seed
controlável e com dropout 0.0, o pacote **não pode** servir para reproduzir Tab. 3 nem
para revisão por pares.

**Fix sugerido:**
1. `_build_args`: `seed=self.seed` (não `randint`).
2. Remover `th.seed()` do print; logar `self.args.seed` apenas.
3. Seedar `random`, `numpy`, `torch` (e `cuda` se houver) no início de preprocess e treino.
4. `GCN(..., dropout_rate=self.args.dropout)`.

**Como validar:**
```bash
# Dois runs com mesmo seed devem coincidir (tol. numérica mínima)
python -c "from text4gcn import GNN; ..."  # script de regressão — ver #16
pytest tests/ -k seed -q --no-cov   # adicionar teste dedicado
```

---

### 0b. Early-stop avalia o modelo errado ✅

**Arquivo:** `models/train_eval.py` (`_train`).

**Evidência:** ao disparar early stopping, o código salva o `state_dict` **do epoch que
piorou** e chama `_test` nesse modelo — não no melhor val loss. O paper Yao / prática
padrão usam checkpoint do melhor validation.

**Fix sugerido:** manter `best_state` / `best_val_loss`; no stop ou fim do loop, carregar
`best_state` antes de `_test`.

**Como validar:** teste unitário com val loss artificialmente não-monotônica; assert que
o modelo testado corresponde ao melhor val.

---

### 0c. `GNN(val_ratio=…)` é no-op ✅

**Arquivos:** `models/gnn.py`, `preprocess/pipeline.py`.

O split de validação é materializado em `TextPipeline.execute(validation_ratio=…)` ao
escrever `x/y/...`. O parâmetro `val_ratio` do `GNN` é copiado para `args` mas **não
altera** máscaras já salvas.

**Fix sugerido (escolher um):**
- (a) Remover `val_ratio` de `GNN` e documentar que o split vive no preprocess; ou
- (b) Falhar com erro claro se `val_ratio` ≠ ratio usado nos arquivos (ler metadado); ou
- (c) Recomputar máscaras em `fit()` a partir de `val_ratio` (mudança de semântica — documentar).

Alinhado ao paper: §3.2 diz “10% of the training set as validation” — deve ser
**explícito e único** na API.

---

### 0d. Fidelidade RWF: peso das arestas ≠ paper ✅

**Arquivos:** `modules/adjacency.py` (`compute_weights_with_PMI`), builders em
`builder/{word2vec,bert,dependency,linguistic_inquiry,_adj}.py`.

**Paper (Eq. 6–8):** RWF decide se a aresta existe (filtro sintático/semântico/linguístico);
depois PMI > 0 como `Rdep`; peso word–word é normalização do tipo
`O(wi,wj) / T(wi,wj)` (Max-Min / ocorrência relativa) — **não** z-score de contagens.

**Código atual (`compute_weights_with_PMI`):** PMI só como *gate* (`if pmi <= 0: continue`);
o peso gravado é `(count - mean) / std` (z-score). Isso diverge do paper e de Yao (que
grava o **valor PMI** no path Frequency).

**Além disso:** no path Frequency, `extract_pmi_word_weights` está **correto** (grava PMI).
A inconsistência é entre o baseline TGCN e os builders RWF — exatamente onde o claim
DCAI precisa ser sólido.

**Fix sugerido:**
1. Implementar peso conforme Eq. (6) (ou documentar desvio explícito se houver razão).
2. Separar funções: `gate_by_pmi` vs `edge_weight_rwf` vs `edge_weight_pmi` (Yao).
3. Testes que fixam um mini-corpus e comparam #arestas / pesos com valores gold.
4. Não chamar o helper de “PMI weights” se ele não armazena PMI.

**Como validar:** reproduzir ordem de grandeza da Tab. 2 em R8/R52 (ver #17).

---

## FASE P0 — Bugs do path publicado

### 1. Bug: `apply_refinement` quebra com `dia_matrix` ✅

**Arquivo:** `text4gcn/graph/refinement.py` (~linha 290).

```
TypeError: 'dia_matrix' object is not subscriptable
features_sub = features[keep_indices, :]
```

**Causa:** `sp.issparse` é True para `dia_matrix`, que não suporta fancy indexing.

**Fix:**
```python
if sp.issparse(features):
    features_sub = sp.csr_matrix(features)[keep_indices, :]
else:
    features_sub = np.asarray(features)[keep_indices, :]
```

**Relevância científica:** `refinement` estende o claim de sparsificação/memória do paper.
Feature quebrada = claim de engenharia incompleto.

**Como validar:** `pytest tests/test_graph_refinement.py -q --no-cov`

---

### 2. Bug: IDs trocados no `AsyncProcessor`

**Arquivo:** `optimization/async_operations.py` — prefixo `task_` duplicado.

**Nota:** módulo **órfão** (ver #6). **Não corrigir** — remover com o pacote `optimization/`.
Mantido no plano só para não surpreender quem olha testes falhando.

---

### 3. `GNN`/`TextPipeline` não validam no `__init__` ✅

Testes esperam falha rápida com dataset/path inválidos; hoje a falha ocorre tarde (I/O
genérico). `GNN.fit` já valida paths de node_features/adjacency — alinhar `__init__` ou
ajustar testes à política lazy documentada. Cuidado para não quebrar o fluxo
“criar pipeline → gerar arquivos”.

---

### 4. `Layer` inconsistente — API promete GAT ✅

- `layers.py`: `GCN`, `GAT` (constantes).
- `gnn.py`: rejeita tudo ≠ `Layer.GCN`.
- `config_manager.py` (órfão): aceita GCN e GAT.
- **Paper DCAI:** experimentos só com GCN 2 camadas.

**Fix A0/P0:** remover `GAT` do enum público até existir implementação + experimento.
Não prometer o que o paper não avalia.

---

## FASE A1 — Âncora no paper (reprodução e honestidade)

### 17. Suite de reprodução das Tabelas 2/3 do paper ✅

**Objetivo:** script(s) + docs que, com datasets públicos do paper, reportam:
- número de arestas por builder (Tab. 2);
- test accuracy média ± std em N seeds (Tab. 3), com protocolo §3.2.

**Mínimo viável:** R8 + R52, builders Frequency (TGCN) e Semantic; comparar ordem de
grandeza de arestas e accuracy vs Tab. 2/3 (tolerância documentada para diferenças de
limpeza NLTK / versões).

**Por que:** Galke & Scherp e a comunidade pedem reprodução; sem isso o pacote é “código
inspirado no paper”, não artefato do paper.

**Como validar:** `python scripts/reproduce_dcai2022.py --dataset R8 --builders frequency,semantic`

---

### 18. Métricas de arestas/memória expostas na API ✅

O claim central é **redução de arestas**. Hoje o usuário não vê isso de forma first-class.

**Fix sugerido:** após cada builder, logar/retornar `n_nodes`, `n_edges`, `nnz`, tamanho
do arquivo `.adj`; opcionalmente comparar com baseline Frequency no mesmo dataset.

---

### 19. Documentar limitação transdutiva e nicho ✅

README / docs devem declarar explicitamente:

1. Modelo **transdutivo** (test docs entram no grafo) — como Yao e DCAI.
2. Nicho: **text-graph construction + TGCN filtrado (RWF)**; não substituto de
   `transformers` fine-tune para SOTA de textos curtos.
3. Quando grafos ainda ajudam: poucos rótulos, textos longos, estudo de estrutura.
4. Citação canônica: Pereira & de Castro (DCAI 2022) + Yao et al. (AAAI 2019).

---

## FASE P1 — Remover peso morto

### 5. 965 linhas de trainer composto nunca conectado

Arquivos: `textgcn_trainer.py`, `config_manager.py`, `data_processor.py`,
`model_factory.py`, `model_evaluator.py`, `model_trainer.py`, `metrics_visualizer.py`.

Path real: `GNN.fit` → `train_eval.TextGCNTrainer`.

**Recomendação:** **remover** (opção a). Não migrar — a versão composta nunca foi validada
nas Tab. 2/3 e diverge (ex.: `epoches`, aceita GAT).

---

### 6. 5.930 linhas de `di/`, `optimization/`, `services/` não usadas

~56% do LOC; 0% do método científico do paper; ~69% do flake8.

**Recomendação:** remover do pacote publicado (branch experimental se quiser preservar).
Antítese do foco DCAI (grafo enxuto) — infra genérica que ninguém usa.

---

### 7. Testes obsoletos do refactor anterior

`tests/test_refactoring_simple.py`, `test_refactored_models.py`,
`test_refactored_preprocessing.py` — imports mortos (`text_pipeline_refactored`, etc.).

Apagar junto com #5/#6.

---

### 8. Lint flake8

~1698 ocorrências; maioria some com #6 + `black`/`isort`. Depois: F401, F841, E722, F821.

---

### 9. Artefatos de build versionados no git

`docs/_site/`, `docs/.jekyll-cache/` — adicionar ao `.gitignore` e `git rm --cached`.

---

## FASE P2 — Tipagem

### 10. mypy

474 erros; maioria em órfãos. Remedir **depois** de #5/#6. No path vivo, tipar
`Optional` onde `model`/`data` podem ser None.

### 11. Typo `epoches`

Só no trainer órfão. Some com #5. API pública `GNN` já usa `epochs`.

---

## FASE P3 — Fricção vs módulos do paper

### 12. Stanford CoreNLP — fricção vs módulo Sintático do paper

O paper **depende** de CoreNLP para o filtro SYNTACTIC. Remover a dependência sem
substituto **apaga** um dos três pilares RWF.

**Estratégia alinhada ao paper:**
1. Extra opcional: `pip install text4gcn[syntactic]` (Java + path do jar documentado); **ou**
2. Backend alternativo spaCy com teste de paridade de arestas vs CoreNLP em corpus mínimo;
3. Frequency + Semantic (Word2Vec) devem funcionar com `pip install text4gcn` puro.

LIWC continua proprietário (paper já avisa) — documentar e manter opcional.

---

### 13. CONTRIBUTING.md duplicado

Fonte única na raiz; docs só linka.

---

## FASE P4 — Extensões (só após A0–A1)

### 14. Híbrido BertGCN — opcional; não é o claim do paper

BertGCN (Lin et al. 2021) ganha 1–3 pp vs TextGCN puro nos benchmarks clássicos, mas
**muda a pergunta científica** (híbrido PLM+GNN vs RWF/memória).

Só implementar se:
- A0–A1 estiverem sólidos; e
- for nomeado `BertGCN` / `hybrid=True` com citação correta — **nunca** confundir com
  `Semantic.bert()` (construção de arestas).

Esforço grande; retorno de marketing médio; retorno para o paper DCAI **baixo**.

---

### 15. Implementar GAT de verdade ou remover do enum

Paper não avalia GAT. Preferência: **remover do enum** (barato, honesto). Implementar
só com experimento e baseline reportado.

---

## FASE P5 — Sustentar o ganho

### 16. CI: reprodutibilidade + lint

Após baseline limpa:
```yaml
- run: pytest tests/ -q
- run: flake8 text4gcn --max-line-length=100 --count --max-count=0  # ajustar teto
- run: mypy text4gcn --ignore-missing-imports
- run: pytest tests/test_reproducibility_seed.py  # mesmo seed → mesma accuracy
- run: pytest tests/test_graph_refinement.py      # regressão dia_matrix
```

---

### 20. Pesquisa futura alinhada ao paper: inductive / subgraphs

O próprio DCAI 2022 lista: embeddings mais ricos; **subgrafos** para escala. Direções
compatíveis com SOTA (InducT-GCN, sampling PyG/DGL) sem abandonar o nicho RWF.

Tratar como **roadmap de pesquisa**, não como bug do repositório.

---

## Ordem de execução recomendada (revisada)

```
A0  0a → 0b → 0c → 0d     rigor: seed/dropout/early-stop/val + peso RWF
P0  1, 3, 4               bugs path publicado (skip 2 → remover no 6)
A1  17, 18, 19            âncora no paper + docs honestos
P1  6 → 5 → 7 → 8 → 9     enxugar (6 antes de 5)
P2  10, 11                tipagem no que sobrou
P3  12, 13                CoreNLP como extra; CONTRIBUTING
P4  15 (remover GAT) → 14 (BertGCN só se houver tese clara)
P5  16; 20 como pesquisa
```

**Não fazer nesta ordem o inverso do plano antigo** (BertGCN/GAT antes de
reprodutibilidade). Competir com transformers sem Tab. 2/3 reproduzíveis enfraquece
o único claim que o pacote tem no SOTA atual: **grafos de texto filtrados e eficientes**.

---

## Checklist de “pacote academicamente defensável”

- [x] `GNN(seed=k)` é determinístico end-to-end (preprocess + train) — *seed wired; validação e2e completa fica no #16*
- [x] Dropout 0.5 efetivamente aplicado (como §3.2)
- [x] Test accuracy usa weights do melhor val
- [x] Builders RWF implementam Eq. (6)–(8) ou documentam desvio
- [x] Script reproduz ordem de grandeza Tab. 2 e accuracy Tab. 3 em R8 — *MVP: referência + compare adj; treino via `--train`*
- [x] README declara nicho (RWF/memória) e limitação transdutiva
- [x] Zero código órfão no wheel publicado — *di/optimization/services + trainer composto removidos*
- [x] CI impede regressão de seed e de `dia_matrix` — *pytest A0 + refinement no workflow*

