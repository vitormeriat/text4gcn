#!/usr/bin/env python3
"""
Script para configurar ambiente virtual com versão específica do Python.
Uso: python setup_venv.py [versão]
Exemplo: python setup_venv.py 3.12
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def check_python_version(version):
    """Verifica se a versão do Python está disponível."""
    try:
        result = subprocess.run(
            [f"python{version}", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Python {version} encontrado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"❌ Python {version} não encontrado.")
        print(f"💡 Instale com: pyenv install {version}")
        return False


def create_virtual_environment(version, venv_name=".venv"):
    """Cria ambiente virtual com a versão especificada."""
    print(f"🔧 Criando ambiente virtual com Python {version}...")
    
    # Remover ambiente virtual existente se houver
    if Path(venv_name).exists():
        print(f"🗑️  Removendo ambiente virtual existente: {venv_name}")
        import shutil
        shutil.rmtree(venv_name)
    
    # Criar novo ambiente virtual
    cmd = [f"python{version}", "-m", "venv", venv_name]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Ambiente virtual criado: {venv_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao criar ambiente virtual: {e}")
        return False


def install_dependencies(venv_name=".venv"):
    """Instala as dependências no ambiente virtual."""
    print("📦 Instalando dependências...")
    
    # Determinar comando pip baseado no OS
    if os.name == 'nt':  # Windows
        pip_cmd = f"{venv_name}\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = f"{venv_name}/bin/pip"
    
    # Atualizar pip
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Instalar dependências
    try:
        # Instalar em modo de desenvolvimento com dependências CPU
        subprocess.run([pip_cmd, "install", "-e", ".[cpu]"], check=True)
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False


def create_activation_script(version, venv_name=".venv"):
    """Cria script de ativação personalizado."""
    script_content = f"""#!/bin/bash
# Script de ativação para Text4GCN com Python {version}
echo "🐍 Ativando Text4GCN com Python {version}..."
source {venv_name}/bin/activate
echo "✅ Ambiente ativado!"
echo "💡 Para desativar: deactivate"
"""
    
    script_path = Path("activate_text4gcn.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Tornar executável
    os.chmod(script_path, 0o755)
    print(f"📝 Script de ativação criado: {script_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Configurar ambiente virtual para Text4GCN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python setup_venv.py 3.10    # Usar Python 3.10
  python setup_venv.py 3.12    # Usar Python 3.12
  python setup_venv.py 3.13    # Usar Python 3.13
        """
    )
    
    parser.add_argument(
        "version",
        nargs="?",
        default="3.11",
        help="Versão do Python a usar (padrão: 3.11)"
    )
    
    parser.add_argument(
        "--venv-name",
        default=".venv",
        help="Nome do ambiente virtual (padrão: .venv)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Configurando Text4GCN")
    print("="*50)
    print(f"Versão do Python: {args.version}")
    print(f"Ambiente virtual: {args.venv_name}")
    print("="*50)
    
    # Verificar se a versão do Python está disponível
    if not check_python_version(args.version):
        sys.exit(1)
    
    # Criar ambiente virtual
    if not create_virtual_environment(args.version, args.venv_name):
        sys.exit(1)
    
    # Instalar dependências
    if not install_dependencies(args.venv_name):
        sys.exit(1)
    
    # Criar script de ativação
    create_activation_script(args.version, args.venv_name)
    
    print("\n" + "="*50)
    print("🎉 Configuração concluída!")
    print("="*50)
    print(f"Para ativar o ambiente:")
    print(f"  source {args.venv_name}/bin/activate")
    print(f"  # ou")
    print(f"  ./activate_text4gcn.sh")
    print(f"\nPara testar a instalação:")
    print(f"  python -c \"import text4gcn; print('Text4GCN instalado com sucesso!')\"")
    print(f"  pytest tests/ -v")


if __name__ == "__main__":
    main()
