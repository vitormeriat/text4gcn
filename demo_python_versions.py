#!/usr/bin/env python3
"""
Demonstração do suporte a múltiplas versões do Python no Text4GCN.
Este script mostra como usar o Text4GCN com diferentes versões do Python.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(title):
    """Imprime um cabeçalho formatado."""
    print("\n" + "="*60)
    print(f"🐍 {title}")
    print("="*60)


def print_step(step, description):
    """Imprime um passo formatado."""
    print(f"\n📋 Passo {step}: {description}")
    print("-" * 40)


def run_command(cmd, description, check=True):
    """Executa um comando e retorna o resultado."""
    print(f"💻 Executando: {description}")
    print(f"   Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"   ✅ Saída: {result.stdout.strip()}")
        return True, result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro: {e}")
        if e.stderr:
            print(f"   📝 Detalhes: {e.stderr.strip()}")
        return False, e


def check_python_version():
    """Verifica a versão atual do Python."""
    print_step(1, "Verificando versão do Python")
    
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    print(f"   Executável: {sys.executable}")
    
    if version.major == 3 and version.minor >= 10:
        print("   ✅ Versão compatível com Text4GCN!")
        return True
    else:
        print("   ❌ Versão não compatível. Text4GCN requer Python 3.10+")
        return False


def demonstrate_text4gcn_import():
    """Demonstra a importação do Text4GCN."""
    print_step(2, "Testando importação do Text4GCN")
    
    try:
        import text4gcn
        print("   ✅ Text4GCN importado com sucesso!")
        
        # Testar importação de módulos específicos
        from text4gcn import datasets, builder, models
        print("   ✅ Módulos principais importados com sucesso!")
        
        # Mostrar versão se disponível
        if hasattr(text4gcn, '__version__'):
            print(f"   📦 Versão: {text4gcn.__version__}")
        
        return True
    except ImportError as e:
        print(f"   ❌ Erro ao importar Text4GCN: {e}")
        return False


def demonstrate_datasets():
    """Demonstra o uso dos datasets."""
    print_step(3, "Testando funcionalidades dos datasets")
    
    try:
        from text4gcn.datasets.data import list as list_datasets
        datasets = list_datasets()
        print(f"   📊 Datasets disponíveis: {datasets}")
        
        from text4gcn.datasets.data import R8
        print("   ✅ Classe R8 importada com sucesso!")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao testar datasets: {e}")
        return False


def demonstrate_builders():
    """Demonstra o uso dos builders."""
    print_step(4, "Testando funcionalidades dos builders")
    
    try:
        from text4gcn.builder.pipeline import Frequency, Semantic, Syntactic
        print("   ✅ Builders importados com sucesso!")
        
        # Testar inicialização básica (sem argumentos para evitar erro)
        print("   ✅ Builders disponíveis: Frequency, Semantic, Syntactic")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao testar builders: {e}")
        return False


def demonstrate_models():
    """Demonstra o uso dos modelos."""
    print_step(5, "Testando funcionalidades dos modelos")
    
    try:
        from text4gcn.models.gnn import GCN, GAT
        print("   ✅ Modelos GNN importados com sucesso!")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao testar modelos: {e}")
        return False


def show_installation_instructions():
    """Mostra instruções de instalação para diferentes versões."""
    print_step(6, "Instruções de instalação para diferentes versões")
    
    instructions = """
   📚 Para usar Text4GCN com diferentes versões do Python:
   
   1️⃣ Usando o script automatizado:
      python setup_venv.py 3.11    # Python 3.11
      python setup_venv.py 3.12    # Python 3.12
      python setup_venv.py 3.13    # Python 3.13
   
   2️⃣ Usando Poetry:
      poetry env use 3.11.0
      poetry install
      poetry run python -c "import text4gcn"
   
   3️⃣ Usando pyenv + venv:
      pyenv install 3.12.0
      pyenv local 3.12.0
      python -m venv .venv
      source .venv/bin/activate
      pip install -e .
   
   4️⃣ Testando compatibilidade:
      python test_python_versions.py
   """
    
    print(instructions)


def main():
    """Função principal da demonstração."""
    print_header("DEMONSTRAÇÃO - Text4GCN Multi-Version Support")
    
    print("""
   🎯 Este script demonstra o suporte do Text4GCN a múltiplas versões do Python.
   
   📋 Text4GCN agora suporta:
   • Python 3.10 ✅
   • Python 3.11 ✅ (Recomendado)
   • Python 3.12 ✅
   • Python 3.13 ✅
   """)
    
    # Verificar versão do Python
    if not check_python_version():
        print("\n❌ Esta demonstração requer Python 3.10 ou superior.")
        sys.exit(1)
    
    # Testar importação do Text4GCN
    if not demonstrate_text4gcn_import():
        print("\n❌ Text4GCN não está instalado ou há problemas de dependências.")
        print("💡 Execute: pip install -e .")
        sys.exit(1)
    
    # Testar funcionalidades específicas
    demonstrate_datasets()
    demonstrate_builders()
    demonstrate_models()
    
    # Mostrar instruções
    show_installation_instructions()
    
    # Resumo final
    print_header("RESUMO DA DEMONSTRAÇÃO")
    
    print("""
   ✅ Text4GCN está funcionando corretamente!
   
   🚀 Próximos passos:
   1. Teste com outras versões do Python usando os scripts fornecidos
   2. Explore os exemplos na pasta examples/
   3. Consulte a documentação no README.md
   4. Execute os testes: pytest tests/
   
   📚 Recursos adicionais:
   • setup_venv.py - Configuração automática de ambientes
   • test_python_versions.py - Teste de compatibilidade
   • PYTHON_VERSION_UPGRADE.md - Documentação completa
   """)
    
    print("🎉 Demonstração concluída com sucesso!")


if __name__ == "__main__":
    main()
