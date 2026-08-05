#!/usr/bin/env python3
"""
Script de configuração automática do ambiente Text4GCN
Alternativa Python para setup_env.sh
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """Cores para output no terminal"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_status(message):
    """Imprimir mensagem de status"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def print_success(message):
    """Imprimir mensagem de sucesso"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def print_warning(message):
    """Imprimir mensagem de aviso"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message):
    """Imprimir mensagem de erro"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def run_command(command, check=True, capture_output=False):
    """Executar comando e retornar resultado"""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True, check=check)
            return True, "", ""
    except subprocess.CalledProcessError as e:
        return False, "", str(e)


def check_python_version():
    """Verificar versão do Python"""
    print_status("Verificando versão do Python...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} encontrado")
        return True
    else:
        print_error(f"Python 3.10+ é necessário. Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False


def check_poetry():
    """Verificar se Poetry está instalado"""
    print_status("Verificando Poetry...")
    
    success, stdout, stderr = run_command("poetry --version", capture_output=True)
    if success:
        print_success(f"Poetry encontrado: {stdout.strip()}")
        return True
    else:
        print_error("Poetry não encontrado. Instale Poetry primeiro:")
        print_status("curl -sSL https://install.python-poetry.org | python3 -")
        return False


def clean_environment():
    """Limpar ambiente anterior"""
    print_status("Limpando ambiente anterior...")
    
    # Remover .venv se existir
    venv_path = Path(".venv")
    if venv_path.exists():
        print_warning("Removendo .venv existente...")
        import shutil
        shutil.rmtree(venv_path)
    
    # Remover .python-version se existir
    python_version_file = Path(".python-version")
    if python_version_file.exists():
        print_warning("Removendo .python-version existente...")
        python_version_file.unlink()


def setup_poetry():
    """Configurar Poetry"""
    print_status("Configurando Poetry...")
    
    # Configurar Poetry para usar Python atual
    success, _, _ = run_command("poetry env use python")
    if success:
        print_success("Poetry configurado para usar Python atual")
    else:
        print_warning("Falha ao configurar Poetry - tentando continuar...")


def install_dependencies():
    """Instalar dependências"""
    print_status("Instalando dependências...")
    
    success, _, stderr = run_command("poetry install", capture_output=True)
    if success:
        print_success("Dependências instaladas com sucesso!")
        return True
    else:
        print_error(f"Falha ao instalar dependências: {stderr}")
        print_status("Tentando instalar manualmente...")
        
        # Tentar instalar com --no-cache
        success, _, stderr = run_command("poetry install --no-cache", capture_output=True)
        if success:
            print_success("Dependências instaladas com sucesso!")
            return True
        else:
            print_error(f"Falha ao instalar dependências: {stderr}")
            return False


def test_installation():
    """Testar instalação"""
    print_status("Testando instalação...")
    
    # Testar importação do Text4GCN
    success, stdout, stderr = run_command(
        "poetry run python -c \"import text4gcn; print(f'Text4GCN {text4gcn.__version__} instalado com sucesso!')\"",
        capture_output=True
    )
    
    if success:
        print_success(stdout.strip())
        return True
    else:
        print_error(f"Falha na instalação do Text4GCN: {stderr}")
        return False


def test_pytorch():
    """Testar PyTorch"""
    print_status("Verificando PyTorch...")
    
    success, stdout, stderr = run_command(
        "poetry run python -c \"import torch; print(f'PyTorch {torch.__version__} instalado')\"",
        capture_output=True
    )
    
    if success:
        print_success(stdout.strip())
        
        # Verificar CUDA
        success, stdout, stderr = run_command(
            "poetry run python -c \"import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')\"",
            capture_output=True
        )
        
        if success:
            if "True" in stdout:
                print_success("CUDA disponível!")
            else:
                print_warning("CUDA não disponível - usando CPU")
        
        return True
    else:
        print_warning("PyTorch não encontrado - instalando...")
        success, _, stderr = run_command(
            "poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            capture_output=True
        )
        
        if success:
            print_success("PyTorch instalado com sucesso!")
            return True
        else:
            print_error(f"Falha ao instalar PyTorch: {stderr}")
            return False


def test_synthetic_data():
    """Testar com dados sintéticos"""
    print_status("Testando com dados sintéticos...")
    
    # Ir para diretório de exemplos
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print_warning("Diretório examples não encontrado - pulando teste")
        return True
    
    # Executar teste simples
    success, stdout, stderr = run_command(
        "poetry run python simple_synthetic_test.py",
        capture_output=True,
        check=False
    )
    
    if success:
        print_success("Teste com dados sintéticos passou!")
        return True
    else:
        print_warning("Teste com dados sintéticos falhou - mas a instalação básica está OK")
        return True


def main():
    """Função principal"""
    print("🚀 Configurando ambiente Text4GCN...")
    print(f"📋 Sistema: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print()
    
    # Verificar se estamos no diretório correto
    if not Path("pyproject.toml").exists():
        print_error("Execute este script no diretório raiz do projeto Text4GCN")
        sys.exit(1)
    
    # Verificações
    if not check_python_version():
        sys.exit(1)
    
    if not check_poetry():
        sys.exit(1)
    
    # Configuração
    clean_environment()
    setup_poetry()
    
    if not install_dependencies():
        print_error("Falha na instalação das dependências")
        sys.exit(1)
    
    if not test_installation():
        print_error("Falha no teste de instalação")
        sys.exit(1)
    
    test_pytorch()
    test_synthetic_data()
    
    # Resumo final
    print()
    print_success("🎉 Ambiente configurado com sucesso!")
    print()
    print_status("📋 Próximos passos:")
    print("  1. Ativar ambiente: poetry shell")
    print("  2. Testar instalação: python -c \"import text4gcn; print(text4gcn.__version__)\"")
    print("  3. Executar exemplos: cd examples && python simple_synthetic_test.py")
    print("  4. Abrir notebook: jupyter notebook synthetic_data_demo.ipynb")
    print()
    print_status("💡 Comandos úteis:")
    print("  - poetry shell          # Ativar ambiente")
    print("  - poetry run python     # Executar Python no ambiente")
    print("  - poetry install        # Instalar/atualizar dependências")
    print("  - poetry update         # Atualizar dependências")
    print()
    print_success("Ambiente pronto para uso! 🚀")


if __name__ == "__main__":
    main()
