# ✅ Checklist de Validação - Text4GCN Multi-Version Support

## 🎯 Checklist de Validação Completa

Use este checklist para verificar se todas as funcionalidades estão funcionando corretamente.

### 📋 Configuração Básica

- [ ] **Python 3.10 instalado**
  ```bash
  pyenv install 3.10.0
  pyenv versions | grep 3.10.0
  ```

- [ ] **Python 3.11 instalado**
  ```bash
  pyenv install 3.11.0
  pyenv versions | grep 3.11.0
  ```

- [ ] **Python 3.12 instalado**
  ```bash
  pyenv install 3.12.0
  pyenv versions | grep 3.12.0
  ```

- [ ] **Python 3.13 instalado**
  ```bash
  pyenv install 3.13.6
  pyenv versions | grep 3.13.6
  ```

### 🧪 Testes de Instalação

#### Teste 1: Python 3.10
```bash
pyenv local 3.10.0
python setup_venv.py 3.10
source .venv/bin/activate
python -c "import text4gcn; print('Python 3.10: ✅')"
```

#### Teste 2: Python 3.11
```bash
pyenv local 3.11.0
python setup_venv.py 3.11
source .venv/bin/activate
python -c "import text4gcn; print('Python 3.11: ✅')"
```

#### Teste 3: Python 3.12
```bash
pyenv local 3.12.0
python setup_venv.py 3.12
source .venv/bin/activate
python -c "import text4gcn; print('Python 3.12: ✅')"
```

#### Teste 4: Python 3.13
```bash
pyenv local 3.13.6
python setup_venv.py 3.13
source .venv/bin/activate
python -c "import text4gcn; print('Python 3.13: ✅')"
```

### 🔧 Testes de Funcionalidade

#### Teste de Importação
```bash
python -c "
import text4gcn
from text4gcn.datasets.data import R8, R52, AG_NEWS
from text4gcn.builder.pipeline import Frequency, Semantic, Syntactic
from text4gcn.models.gnn import GCN
print('✅ Todas as importações funcionando!')
"
```

#### Teste de Datasets
```bash
python -c "
from text4gcn.datasets.data import list as list_datasets
datasets = list_datasets()
print(f'📊 Datasets: {datasets}')
print('✅ Datasets funcionando!')
"
```

#### Teste de Builders
```bash
python -c "
from text4gcn.builder.pipeline import Frequency, Semantic, Syntactic
print('✅ Builders: Frequency, Semantic, Syntactic')
print('✅ Builders funcionando!')
"
```

#### Teste de Modelos
```bash
python -c "
from text4gcn.models.gnn import GCN
print('✅ Modelo GCN disponível')
print('✅ Modelos funcionando!')
"
```

### 🚀 Testes de Scripts

#### Script de Configuração
```bash
# Testar com cada versão
python setup_venv.py 3.10
python setup_venv.py 3.11
python setup_venv.py 3.12
python setup_venv.py 3.13
```

#### Script de Teste
```bash
python test_python_versions.py
```

#### Script de Demonstração
```bash
python demo_python_versions.py
```

### 📚 Testes de Documentação

- [ ] **README.md atualizado**
  - Seção "Python Version Support" presente
  - Instruções para múltiplas versões
  - Badge atualizado para 3.10-3.13

- [ ] **PYTHON_VERSION_UPGRADE.md completo**
  - Guia detalhado de atualização
  - Instruções de uso
  - Solução de problemas

- [ ] **Scripts documentados**
  - `setup_venv.py` com help
  - `test_python_versions.py` com instruções
  - `demo_python_versions.py` funcional

### 🔄 Testes de CI/CD

#### GitHub Actions
- [ ] **Workflow python-matrix.yml**
  - Testa Python 3.10, 3.11, 3.12, 3.13
  - Executa testes de linting
  - Executa build do pacote

#### Teste Local do Workflow
```bash
# Simular teste do GitHub Actions
python -m pytest tests/ -v
python -m black --check .
python -m isort --check-only .
python -m flake8 .
python -m mypy text4gcn/
```

### 🐛 Testes de Problemas Conhecidos

#### Problema: "pyenv: version not installed"
```bash
# Solução
pyenv install 3.10.0
pyenv install 3.11.0
pyenv install 3.12.0
pyenv install 3.13.6
```

#### Problema: "ModuleNotFoundError: No module named 'torch'"
```bash
# Solução
poetry run pip install torch
# ou
pip install torch
```

#### Problema: "ModuleNotFoundError: No module named 'requests'"
```bash
# Solução
poetry install
# ou
pip install -e .
```

### 📊 Relatório Final

#### Status por Versão
- [ ] **Python 3.10**: ✅ Funcionando
- [ ] **Python 3.11**: ✅ Funcionando
- [ ] **Python 3.12**: ✅ Funcionando
- [ ] **Python 3.13**: ✅ Funcionando

#### Status por Funcionalidade
- [ ] **Importação**: ✅ Funcionando
- [ ] **Datasets**: ✅ Funcionando
- [ ] **Builders**: ✅ Funcionando
- [ ] **Modelos**: ✅ Funcionando
- [ ] **Scripts**: ✅ Funcionando
- [ ] **Documentação**: ✅ Completa
- [ ] **CI/CD**: ✅ Configurado

### 🎯 Critérios de Sucesso

Para considerar o upgrade como **BEM-SUCEDIDO**, todos os itens abaixo devem estar marcados:

- [ ] **4 versões do Python** (3.10, 3.11, 3.12, 3.13) funcionando
- [ ] **Scripts automatizados** funcionando
- [ ] **Testes de compatibilidade** passando
- [ ] **Documentação completa** e atualizada
- [ ] **CI/CD configurado** e funcionando
- [ ] **Sem breaking changes** para usuários existentes

### 🚨 Problemas Encontrados

Se algum teste falhar, documente aqui:

```
Versão: ___________
Problema: ___________
Solução: ___________
Status: ___________
```

### ✅ Assinatura de Validação

- [ ] **Testado por**: ________________
- [ ] **Data**: ________________
- [ ] **Versões validadas**: 3.10, 3.11, 3.12, 3.13
- [ ] **Status final**: ✅ APROVADO / ❌ REPROVADO

---

**Nota**: Este checklist deve ser executado completamente antes de considerar o upgrade como finalizado.

