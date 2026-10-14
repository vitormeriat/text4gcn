# Tutorial: publicar, validar e testar treino com GPU (Colab / Azure ML)

> Resposta direta à pergunta: **sim**. O fluxo recomendado é publicar primeiro no **TestPyPI** (índice de testes, separado do PyPI real), instalar essa versão de teste no Colab/Azure ML para validar o treino com GPU, e só depois publicar a versão final no **PyPI**. É exatamente para isso que o TestPyPI existe — nenhuma mudança no código é necessária, só no processo de release.

---

## 0. Validação já feita neste branch (`fix/blockers-and-solid-refactor`)

Antes de qualquer publicação, isto já foi verificado localmente:

| Verificação | Resultado |
|---|---|
| `pytest tests/` | **79 passed**, 0 falhas (cobertura: `utils/device.py` 93%, `models/gnn.py` 71%) |
| `python -m build` (sdist + wheel via `poetry-core`) | OK → `text4gcn-2.0.0.tar.gz` / `.whl` |
| `examples/end_to_end_synthetic.py --docs 60 --epochs 5` | OK → `TextPipeline` → `Frequency.tfidf` → `GNN.fit()` completo, `test_accuracy=1.0000` |

**Ponto de atenção:** `pyproject.toml` define os extras `cpu = []` e `gpu = []` (ambos vazios). `notebooks/PYTORCH_DEVICE_MANAGEMENT.md` descreve esses extras como se instalassem versões distintas do PyTorch, mas hoje `pip install text4gcn[gpu]` não instala nada diferente de `pip install text4gcn` — o torch do ambiente é o que vale. Não é um bug bloqueante (Colab/Azure ML já vêm com torch+CUDA), mas vale corrigir a doc ou implementar os extras de verdade depois.

---

## 1. Fluxo de release: TestPyPI → validação com GPU → PyPI

### 1.1 Por que TestPyPI primeiro

- É um índice PyPI paralelo (`test.pypi.org`), com conta e token **separados** da conta real do PyPI.
- Permite `pip install` a partir de um servidor público real (igual ao Colab faria em produção), sem arriscar publicar uma versão quebrada no índice oficial.
- Uma vez publicado um número de versão no PyPI (ou no TestPyPI), **não dá para reenviar o mesmo número** — por isso a versão de teste deve ter um sufixo de pré-release (`rc`, `dev`, etc.), diferente da versão final.

### 1.2 Configurar o TestPyPI no Poetry (uma vez só)

```bash
# 1. Crie uma conta em https://test.pypi.org (separada da conta do pypi.org)
# 2. Gere um API token em https://test.pypi.org/manage/account/token/

poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi <SEU_TOKEN_TESTPYPI>
```

### 1.3 Publicar a versão de teste

```bash
# marque como pré-release para não colidir com a versão final
poetry version 2.0.1rc1   # ou: poetry version prerelease

poetry build
poetry publish -r testpypi
```

Confirme em: `https://test.pypi.org/project/text4gcn/`

### 1.4 Instalar a versão de teste no Colab/Azure ML

O TestPyPI **não** espelha dependências como `torch`, `transformers`, etc. — por isso é preciso apontar o índice normal como fallback:

```python
!pip install -q \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "text4gcn==2.0.1rc1"
```

Rode o smoke test / treino com GPU (seções 2 e 3 abaixo) usando essa instalação.

### 1.5 Após validar: publicar a versão final

```bash
poetry version 2.0.1        # remove o sufixo rc, versão "de verdade"
poetry build
poetry publish              # PyPI oficial (token pypi-token.pypi já configurado)
```

Ou, se preferir usar o pipeline já existente no repo (`.github/workflows/publish.yml`), crie uma **GitHub Release** com a tag `v2.0.1` — o workflow builda e publica no PyPI automaticamente via `poetry publish` (usa o secret `PYPI_API_TOKEN`). Ele também pode ser disparado manualmente (`workflow_dispatch`) direto na aba Actions.

> Dica: o mesmo workflow pode ser duplicado para TestPyPI (job separado apontando `-r testpypi`), se quiser automatizar também o passo 1.3 via CI em vez de rodar `poetry publish -r testpypi` localmente.

---

## 2. Testar treino com GPU no Google Colab

**1. Ative GPU:** `Ambiente de execução → Alterar tipo de ambiente de execução → GPU (T4)`.

**2. Instalar (versão de teste, enquanto valida):**
```python
!pip install -q \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "text4gcn==2.0.1rc1"
```
*(Depois de validado e publicado no PyPI oficial, troque por `!pip install -q text4gcn==2.0.1`.)*

**3. Validar GPU e device manager:**
```python
import torch
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma")

import text4gcn
print(text4gcn.utils.check_pytorch_installation())
```

**4. Dataset real + treino forçando GPU:**
```python
from pathlib import Path
from text4gcn import GNN, TextPipeline, Builder, Layer
from text4gcn.builder import Frequency
from text4gcn.datasets import data

DATA = Path("data"); DATA.mkdir(exist_ok=True)
data.R8(path=str(DATA))

TextPipeline(dataset_name="R8", dataset_path=str(DATA), seed=1234).execute(validation_ratio=0.10)
Frequency(dataset_name="R8", dataset_path=str(DATA)).tfidf(window_size=20)

result = GNN(
    dataset_name="R8", dataset_path=str(DATA), log_dir="logs",
    builder=Builder.Tfidf, layer=Layer.GCN,
    epochs=200, device="cuda",   # força GPU; use None para auto-detectar
).fit()
print(result)
```
Se funcionar, o `GNN.__init__` imprime `Using device: cuda` + nome/memória da GPU antes de treinar.

**Smoke test rápido (sem baixar dataset real):**
```python
!git clone -b fix/blockers-and-solid-refactor https://github.com/vitormeriat/text4gcn.git
%cd text4gcn
!python examples/end_to_end_synthetic.py --docs 300 --epochs 30
```

---

## 3. Testar treino com GPU no Azure Machine Learning (notebook)

**1. Criar/usar uma compute instance com GPU:** no Studio, `Compute → Compute instances → New`, escolha uma série com GPU (ex.: `Standard_NC6s_v3`). Anexe o notebook a essa instance.

**2. Escolher o kernel:** use um kernel `Python 3.10+` (ex.: `azureml_py310` ou um ambiente `pytorch-x.y-cuda`) — text4gcn exige `>=3.10,<3.14`.

**3. Instalar e validar:**
```python
%pip install -q \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "text4gcn==2.0.1rc1"

import torch, text4gcn
print("CUDA disponível:", torch.cuda.is_available())
print(text4gcn.utils.check_pytorch_installation())
```
Se `cuda_available` vier `False` numa compute GPU, o driver da imagem provavelmente não bate com o torch instalado — reinstale o torch certo antes do text4gcn:
```python
%pip install -q torch --index-url https://download.pytorch.org/whl/cu121
%pip install -q --no-deps text4gcn
```

**4. Rodar o mesmo script de treino da seção 2** (troque `Path("data")` por um caminho no disco anexado ou num Datastore montado, se quiser persistir os dados).

**5. (Opcional) Rodar como Job em vez de notebook interativo:** use `az ml job create` com um `environment` com CUDA, `command: python examples/end_to_end_synthetic.py --epochs 200` — útil para não prender a compute instance interativa durante o treino.

---

## 4. Checklist final

- [ ] `poetry publish -r testpypi` publicado com versão `rcN`/`devN`
- [ ] Instalação via `--index-url test.pypi.org --extra-index-url pypi.org` funcionou no Colab e/ou Azure ML
- [ ] `torch.cuda.is_available() == True`
- [ ] Log do `GNN(...)` mostra `Using device: cuda` + nome da GPU
- [ ] `result.test_accuracy` reportado ao final do `fit()`
- [ ] Tempo por época (`time=...` no log) caiu em relação ao CPU
- [ ] Versão final (sem sufixo `rc`) publicada no PyPI oficial (`poetry publish` ou GitHub Release → `.github/workflows/publish.yml`)
