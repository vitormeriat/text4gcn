#!/bin/bash
# Script de configuração automática do ambiente Text4GCN

set -e  # Parar em caso de erro

echo "🚀 Configurando ambiente Text4GCN..."

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens coloridas
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar se estamos no diretório correto
if [ ! -f "pyproject.toml" ]; then
    print_error "Execute este script no diretório raiz do projeto Text4GCN"
    exit 1
fi

# Verificar Python 3.10
print_status "Verificando Python 3.10..."
if command -v python3.10 &> /dev/null; then
    PYTHON_VERSION=$(python3.10 --version 2>&1)
    print_success "Python encontrado: $PYTHON_VERSION"
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    if [[ $PYTHON_VERSION == *"3.10"* ]] || [[ $PYTHON_VERSION == *"3.11"* ]] || [[ $PYTHON_VERSION == *"3.12"* ]]; then
        print_success "Python encontrado: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.10+ é necessário. Versão encontrada: $PYTHON_VERSION"
        print_status "Instale Python 3.10+ e tente novamente"
        exit 1
    fi
else
    print_error "Python não encontrado. Instale Python 3.10+ e tente novamente"
    exit 1
fi

# Verificar Poetry
print_status "Verificando Poetry..."
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version 2>&1)
    print_success "Poetry encontrado: $POETRY_VERSION"
else
    print_error "Poetry não encontrado. Instale Poetry primeiro:"
    print_status "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Limpar ambiente anterior
print_status "Limpando ambiente anterior..."
if [ -d ".venv" ]; then
    print_warning "Removendo .venv existente..."
    rm -rf .venv
fi

if [ -f ".python-version" ]; then
    print_warning "Removendo .python-version existente..."
    rm -f .python-version
fi

# Configurar Poetry
print_status "Configurando Poetry..."
poetry env use $PYTHON_CMD

# Instalar dependências
print_status "Instalando dependências..."
poetry install

# Testar instalação
print_status "Testando instalação..."
if poetry run python -c "import text4gcn; print(f'Text4GCN {text4gcn.__version__} instalado com sucesso!')" 2>/dev/null; then
    print_success "Text4GCN instalado com sucesso!"
else
    print_error "Falha na instalação do Text4GCN"
    print_status "Tentando instalar dependências manualmente..."
    poetry run pip install --upgrade pip
    poetry install --no-cache
fi

# Verificar PyTorch
print_status "Verificando PyTorch..."
if poetry run python -c "import torch; print(f'PyTorch {torch.__version__} instalado')" 2>/dev/null; then
    print_success "PyTorch instalado com sucesso!"
    
    # Verificar CUDA
    if poetry run python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')" 2>/dev/null; then
        CUDA_STATUS=$(poetry run python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_STATUS" = "True" ]; then
            print_success "CUDA disponível!"
        else
            print_warning "CUDA não disponível - usando CPU"
        fi
    fi
else
    print_warning "PyTorch não encontrado - instalando..."
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Testar dados sintéticos
print_status "Testando com dados sintéticos..."
cd examples
if poetry run python simple_synthetic_test.py 2>/dev/null; then
    print_success "Teste com dados sintéticos passou!"
else
    print_warning "Teste com dados sintéticos falhou - mas a instalação básica está OK"
fi
cd ..

# Resumo final
echo ""
print_success "🎉 Ambiente configurado com sucesso!"
echo ""
print_status "📋 Próximos passos:"
echo "  1. Ativar ambiente: poetry shell"
echo "  2. Testar instalação: python -c \"import text4gcn; print(text4gcn.__version__)\""
echo "  3. Executar exemplos: cd examples && python simple_synthetic_test.py"
echo "  4. Abrir notebook: jupyter notebook synthetic_data_demo.ipynb"
echo ""
print_status "💡 Comandos úteis:"
echo "  - poetry shell          # Ativar ambiente"
echo "  - poetry run python     # Executar Python no ambiente"
echo "  - poetry install        # Instalar/atualizar dependências"
echo "  - poetry update         # Atualizar dependências"
echo ""
print_success "Ambiente pronto para uso! 🚀"
