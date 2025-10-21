# Configuração do Ambiente Local para Text4GCN

Este guia explica como configurar um ambiente virtual para testar e executar o Text4GCN localmente.

## 🐍 **Problema Comum: Conflito de Versões Python**

O erro que você está enfrentando geralmente ocorre quando:
- O Poetry está configurado para usar uma versão específica do Python
- A versão não está instalada no sistema
- Há conflitos entre diferentes gerenciadores de Python (pyenv, conda, system)

## 🛠️ **Soluções**

### **Opção 1: Usar Poetry (Recomendado)**

#### 1.1 Verificar Versão do Python
```bash
# Verificar versões disponíveis
python3 --version
python3.10 --version
python3.11 --version

# Verificar pyenv (se instalado)
pyenv versions
```

#### 1.2 Instalar Python 3.10 (se necessário)
```bash
# Com pyenv
pyenv install 3.10.0
pyenv local 3.10.0

# Ou com apt (Ubuntu/Debian)
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Ou com conda
conda create -n text4gcn python=3.10
conda activate text4gcn
```

#### 1.3 Configurar Poetry
```bash
# Remover configuração atual
rm -rf .venv
rm -f .python-version

# Configurar Poetry para usar Python 3.10
poetry env use python3.10

# Instalar dependências
poetry install

# Ativar ambiente
poetry shell
```

### **Opção 2: Usar venv (Alternativa Simples)**

#### 2.1 Criar Ambiente Virtual
```bash
# Criar ambiente virtual
python3.10 -m venv text4gcn_env

# Ativar ambiente
source text4gcn_env/bin/activate  # Linux/Mac
# ou
text4gcn_env\Scripts\activate     # Windows

# Atualizar pip
pip install --upgrade pip
```

#### 2.2 Instalar Text4GCN
```bash
# Instalar em modo desenvolvimento
pip install -e .

# Ou instalar com dependências
pip install -e .[cpu]  # para CPU
pip install -e .[gpu]  # para GPU
```

### **Opção 3: Usar Conda (Para Ambientes Complexos)**

#### 3.1 Criar Ambiente Conda
```bash
# Criar ambiente
conda create -n text4gcn python=3.10

# Ativar ambiente
conda activate text4gcn

# Instalar dependências
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -e .
```

## 🔧 **Script de Configuração Automática**

Crie um script para automatizar a configuração:

```bash
#!/bin/bash
# setup_env.sh

echo "🚀 Configurando ambiente Text4GCN..."

# Verificar Python
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 não encontrado. Instalando..."
    
    # Tentar instalar com pyenv
    if command -v pyenv &> /dev/null; then
        pyenv install 3.10.0
        pyenv local 3.10.0
    else
        echo "⚠️  Instale Python 3.10 manualmente"
        exit 1
    fi
fi

# Limpar ambiente anterior
echo "🧹 Limpando ambiente anterior..."
rm -rf .venv
rm -f .python-version

# Configurar Poetry
echo "📦 Configurando Poetry..."
poetry env use python3.10
poetry install

# Testar instalação
echo "🧪 Testando instalação..."
poetry run python -c "import text4gcn; print(f'Text4GCN {text4gcn.__version__} instalado com sucesso!')"

echo "✅ Ambiente configurado com sucesso!"
echo "💡 Para ativar o ambiente, execute: poetry shell"
```

## 🧪 **Teste da Instalação**

### **Teste Básico**
```bash
# Ativar ambiente
poetry shell

# Testar importação
python -c "import text4gcn; print(text4gcn.__version__)"

# Testar device management
python -c "from text4gcn.utils.device import check_pytorch_installation; print(check_pytorch_installation())"
```

### **Teste com Dados Sintéticos**
```bash
# Ir para diretório de exemplos
cd examples

# Executar teste simples
python simple_synthetic_test.py

# Ou executar teste completo
python test_textpipeline_synthetic.py
```

## 🐛 **Solução de Problemas**

### **Erro: Python version not found**
```bash
# Verificar versões disponíveis
ls /usr/bin/python*

# Configurar Poetry para usar versão específica
poetry env use /usr/bin/python3.10
```

### **Erro: Permission denied**
```bash
# Dar permissões de execução
chmod +x setup_env.sh

# Ou executar com sudo (não recomendado)
sudo poetry install
```

### **Erro: Package not found**
```bash
# Limpar cache do Poetry
poetry cache clear --all pypi

# Reinstalar
poetry install --no-cache
```

### **Erro: CUDA not available**
```bash
# Instalar versão CPU
poetry install --extras cpu

# Ou instalar PyTorch manualmente
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📋 **Verificação Final**

### **Checklist de Instalação**
- [ ] Python 3.10+ instalado
- [ ] Poetry configurado
- [ ] Ambiente virtual ativado
- [ ] Text4GCN importado com sucesso
- [ ] PyTorch funcionando
- [ ] Testes passando

### **Comandos de Verificação**
```bash
# Verificar Python
python --version

# Verificar Poetry
poetry --version

# Verificar Text4GCN
python -c "import text4gcn; print(text4gcn.__version__)"

# Verificar PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verificar CUDA (se disponível)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 **Execução Rápida**

### **Para Desenvolvimento**
```bash
# 1. Configurar ambiente
poetry install

# 2. Ativar ambiente
poetry shell

# 3. Executar testes
cd examples
python simple_synthetic_test.py
```

### **Para Produção**
```bash
# 1. Instalar versão estável
pip install text4gcn[cpu]  # ou [gpu]

# 2. Testar instalação
python -c "import text4gcn; print('Instalado com sucesso!')"
```

## 📚 **Próximos Passos**

1. **Configurar ambiente** usando uma das opções acima
2. **Testar instalação** com dados sintéticos
3. **Experimentar** com diferentes configurações
4. **Desenvolver** novas funcionalidades
5. **Contribuir** para o projeto

## 💡 **Dicas Importantes**

- **Sempre use Python 3.10+** para compatibilidade
- **Ative o ambiente virtual** antes de executar código
- **Use dados sintéticos** para testes iniciais
- **Verifique logs** em caso de erros
- **Mantenha dependências atualizadas**

Com essas configurações, você deve conseguir executar o Text4GCN localmente sem problemas! 🎉
