# Tutorial: publicar a versão 2.0.0 (TestPyPI → PyPI oficial)

## Sobre a documentação (mkdocs) no TestPyPI/PyPI

**Não.** Nem o TestPyPI nem o PyPI hospedam o site do MkDocs. A página do projeto em qualquer um dos dois índices mostra **apenas** o `long_description` (o `README.md`, conforme `readme = "README.md"` no `pyproject.toml`) renderizado como uma única página estática — sem nav, sem busca, sem as páginas de `docs/guides/*`.

O site de documentação completo (`mkdocs.yml`) é publicado separadamente pelo `.github/workflows/docs.yml`, que roda `mkdocs build --strict` e faz deploy do diretório `site/` para o **GitHub Pages** a cada push em `main`:

- URL: `https://vitormeriat.github.io/text4gcn/`
- Não depende do PyPI/TestPyPI de forma alguma — atualiza sozinho quando você mergear na `main`, independente de quando/se você publicar um pacote.

Ou seja: os dois fluxos são independentes. Publicar no TestPyPI valida o **pacote instalável**; para validar o **site de docs**, faça merge desta branch em `main` (ou rode `mkdocs serve` localmente) e confira o GitHub Pages depois do deploy.

---

## 0. Estado atual (verificado antes de escrever este tutorial)

| Onde | Versões já publicadas |
|---|---|
| PyPI oficial (`pypi.org`) | apenas `1.0.0` |
| TestPyPI (`test.pypi.org`) | `1.0.0` … `1.0.5` |
| `pyproject.toml` (este branch) | `version = "2.0.0"` (ainda não publicado em nenhum dos dois) |

Ou seja: **2.0.0 é de fato a próxima versão real** — não é preciso inventar um número novo, só decidir o sufixo de teste (`2.0.0rc1`) antes de "queimar" o `2.0.0` final.

Este branch (`fix/blockers-and-solid-refactor`) ainda não foi mergeado em `main` — mergear é um pré-requisito antes do release final (ver Passo 1).

---

## 1. Pré-publicação: checklist e preparação

```bash
cd /home/vitor/repos/Text4GNNs/text4gcn

# 1. Testes e build limpos
poetry run pytest tests/ -q
poetry build

# 2. Confirme a versão
poetry version   # deve mostrar 2.0.0
```

**Changelog:** `docs/changelog.md` tem uma seção `## [Unreleased]` com o que mudou nesta versão. Antes do release final, transforme-a em uma seção versionada:

```diff
- ## [Unreleased]
+ ## [2.0.0] - 2026-08-19
```
(mantenha um `## [Unreleased]` vazio acima, para o próximo ciclo).

**Merge para `main`:** abra o PR de `fix/blockers-and-solid-refactor` → `main` e mergeie **antes** de criar a GitHub Release (o workflow `docs.yml` só faz deploy do site em push para `main`; o `publish.yml` roda a partir da tag/release, que normalmente aponta para `main`).

```bash
git push origin fix/blockers-and-solid-refactor
gh pr create --base main --title "Release 2.0.0" --body "Ver docs/changelog.md"
gh pr merge --squash   # ou merge normal, conforme preferir
git checkout main && git pull
```

---

## 2. Configurar credenciais (uma vez só, por máquina)

```bash
# TestPyPI — conta separada em https://test.pypi.org
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi <TOKEN_TESTPYPI>

# PyPI oficial — conta em https://pypi.org
poetry config pypi-token.pypi <TOKEN_PYPI>
```
Gere os tokens em `https://test.pypi.org/manage/account/token/` e `https://pypi.org/manage/account/token/` — escopo "Entire account" na primeira publicação do projeto (depois pode restringir ao projeto `text4gcn`).

> Se preferir publicar via CI em vez de local, o repo já tem `.github/workflows/publish.yml` publicando no PyPI oficial ao criar uma GitHub Release, usando o secret `PYPI_API_TOKEN` (configure em `Settings → Secrets and variables → Actions`).

---

## 3. Publicar a versão de teste no TestPyPI

Na `main`, já mergeada:

```bash
# Marque como pré-release para não gastar o número "2.0.0" no índice de teste
poetry version 2.0.0rc1

rm -rf dist/
poetry build
poetry publish -r testpypi
```

Confirme em: `https://test.pypi.org/project/text4gcn/2.0.0rc1/`
— é aqui que dá pra conferir a página renderizada do README, mas **não** o site de docs (ver seção acima).

### Validar a instalação (em qualquer ambiente limpo — local, Colab, Azure ML)

```bash
pip install -q \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "text4gcn==2.0.0rc1"

python -c "import text4gcn; print(text4gcn.__version__)"
```

Rode o smoke test completo (mesmo usado na validação local deste branch):

```bash
git clone https://github.com/vitormeriat/text4gcn.git /tmp/t4gcn-rc-check
cd /tmp/t4gcn-rc-check
python examples/end_to_end_synthetic.py --docs 300 --epochs 30
```
Esperado: pipeline completo sem erro e `test_accuracy` reportado ao final — mesmo comportamento que validamos localmente (79/79 testes, build OK, `end_to_end_synthetic.py` com `test_accuracy=1.0000`).

Se quiser testar treino com GPU nesse passo, use `notebooks/COLAB_AZURE_GPU_TUTORIAL.md` (trocando o `pip install` de lá pelo comando com `--index-url test.pypi.org` acima).

Encontrou um bug? Corrija o código, suba `2.0.0rc2` (`poetry version 2.0.0rc2` → build → publish) e repita a validação — **nunca reuse `2.0.0rc1`**, o índice não aceita reenvio do mesmo número.

---

## 4. Publicar a versão final (oficial) no PyPI

Depois que a `rcN` validar sem problemas:

```bash
# Remove o sufixo de pré-release — vira a versão real
poetry version 2.0.0

rm -rf dist/
poetry build
poetry publish
```

Ou, usando o pipeline já existente no repo (recomendado, evita publicar do laptop):

```bash
git tag v2.0.0
git push origin v2.0.0
gh release create v2.0.0 --title "v2.0.0" --notes-file <(sed -n '/## \[2.0.0\]/,/## \[/p' docs/changelog.md | sed '$d')
```
Criar a Release dispara `.github/workflows/publish.yml` (`on: release: types: [published]`), que builda e publica no PyPI oficial automaticamente.

---

## 5. Pós-publicação

```bash
# Confirme que instala "de fora", num ambiente limpo, sem apontar índice de teste
pip install -q text4gcn==2.0.0
python -c "import text4gcn; print(text4gcn.__version__)"
```

- Página do pacote: `https://pypi.org/project/text4gcn/2.0.0/`
- Site de docs: já deve estar atualizado em `https://vitormeriat.github.io/text4gcn/` (deploy automático do push para `main` no passo 1 — não depende deste passo de publicação do pacote)
- Badge/versão no `README.md`, se houver, atualizar se estiver hardcoded

---

## Checklist final

- [ ] `pytest tests/` passando, `poetry build` limpo
- [ ] `docs/changelog.md`: `[Unreleased]` → `[2.0.0] - 2026-08-19`
- [ ] Branch mergeado em `main`
- [ ] `poetry version 2.0.0rc1` publicado em TestPyPI
- [ ] Instalação via `--index-url test.pypi.org --extra-index-url pypi.org` validada (import + `end_to_end_synthetic.py`, e opcionalmente GPU no Colab/Azure ML)
- [ ] `poetry version 2.0.0` (final) publicado no PyPI oficial — local ou via GitHub Release
- [ ] `git tag v2.0.0` + GitHub Release criada
- [ ] `pip install text4gcn==2.0.0` funciona num ambiente limpo
- [ ] Site MkDocs no GitHub Pages refletindo a `main` atual
