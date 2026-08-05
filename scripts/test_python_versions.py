#!/usr/bin/env python3
"""
Script para testar o Text4GCN com diferentes versões do Python.
Este script verifica se o projeto é compatível com Python 3.10, 3.11, 3.12 e 3.13.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Executa um comando e retorna o resultado."""
    print(f"\n{'='*60}")
    print(f"Executando: {description}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Sucesso!")
        if result.stdout:
            print("Saída:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Erro!")
        print("Código de saída:", e.returncode)
        if e.stdout:
            print("Saída:", e.stdout)
        if e.stderr:
            print("Erro:", e.stderr)
        return False


def test_python_version(python_version):
    """Testa uma versão específica do Python."""
    print(f"\n🐍 Testando Python {python_version}")
    print("="*50)
    
    # Verificar se a versão está instalada
    version_cmd = [f"python{python_version}", "--version"]
    if not run_command(version_cmd, f"Verificando Python {python_version}"):
        print(f"⚠️  Python {python_version} não está instalado. Pulando...")
        return False
    
    # Criar ambiente virtual
    venv_dir = f".venv-{python_version.replace('.', '')}"
    venv_cmd = [f"python{python_version}", "-m", "venv", venv_dir]
    if not run_command(venv_cmd, f"Criando ambiente virtual com Python {python_version}"):
        return False
    
    # Ativar ambiente virtual e instalar dependências
    if os.name == 'nt':  # Windows
        activate_cmd = f"{venv_dir}\\Scripts\\activate"
        pip_cmd = f"{venv_dir}\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = f"source {venv_dir}/bin/activate"
        pip_cmd = f"{venv_dir}/bin/pip"
    
    # Instalar dependências
    install_cmd = [pip_cmd, "install", "-e", ".", "[cpu]"]
    if not run_command(install_cmd, f"Instalando dependências com Python {python_version}"):
        return False
    
    # Executar testes básicos
    test_cmd = [pip_cmd, "run", "pytest", "tests/", "-v"]
    if not run_command(test_cmd, f"Executando testes com Python {python_version}"):
        print(f"⚠️  Alguns testes falharam com Python {python_version}")
        return False
    
    print(f"✅ Python {python_version} - Todos os testes passaram!")
    return True


def main():
    """Função principal."""
    print("🚀 Testando compatibilidade do Text4GCN com diferentes versões do Python")
    print("="*80)
    
    # Versões do Python para testar
    python_versions = ["3.10", "3.11", "3.12", "3.13"]
    
    results = {}
    
    for version in python_versions:
        results[version] = test_python_version(version)
    
    # Resumo dos resultados
    print("\n" + "="*80)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*80)
    
    for version, success in results.items():
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"Python {version}: {status}")
    
    successful_versions = [v for v, s in results.items() if s]
    failed_versions = [v for v, s in results.items() if not s]
    
    print(f"\n✅ Versões compatíveis: {', '.join(successful_versions) if successful_versions else 'Nenhuma'}")
    print(f"❌ Versões com problemas: {', '.join(failed_versions) if failed_versions else 'Nenhuma'}")
    
    if successful_versions:
        print(f"\n🎉 O Text4GCN é compatível com Python {', '.join(successful_versions)}!")
    else:
        print("\n⚠️  Nenhuma versão do Python testada funcionou corretamente.")
        sys.exit(1)


if __name__ == "__main__":
    main()
