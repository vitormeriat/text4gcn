# 🚀 Configuração Rápida do Ambiente Text4GCN

## ❌ **Problema: Poetry Shell com Erro**

Se você está enfrentando erro com `poetry shell`, aqui estão as soluções:

## ✅ **Soluções Rápidas**

### **1. Script Automático (Recomendado)**

```bash
# Executar script de configuração
./setup_env.sh

# Ou se não funcionar:
python3 setup_env.py
```

### **2. Configuração Manual**

```bash
# 1. Limpar ambiente anterior
rm -rf .venv
rm -f .python-version

# 2. Configurar Poetry
poetry env use python3.10  # ou python3.11, python3.12

# 3. Instalar dependências
poetry install

# 4. Ativar ambiente
poetry shell
```

### **3. Alternativa com venv**

```bash
# 1. Criar ambiente virtual
python3.10 -m venv text4gcn_env

# 2. Ativar ambiente
source text4gcn_env/bin/activate  # Linux/Mac
# text4gcn_env\Scripts\activate   # Windows

# 3. Instalar Text4GCN
pip install -e .
```

## 🧪 **Teste Rápido**

```bash
# Testar instalação
python -c "import text4gcn; print(text4gcn.__version__)"

# Testar com dados sintéticos
cd examples
python simple_synthetic_test.py
```

## 🔧 **Solução de Problemas**

### **Erro: Python version not found**
```bash
# Verificar versões disponíveis
ls /usr/bin/python*

# Usar versão específica
poetry env use /usr/bin/python3.10
```

### **Erro: Permission denied**
```bash
# Dar permissões
chmod +x setup_env.sh

# Executar script
./setup_env.sh
```

### **Erro: Package not found**
```bash
# Limpar cache
poetry cache clear --all pypi

# Reinstalar
poetry install --no-cache
```

## 📋 **Checklist de Verificação**

- [ ] Python 3.10+ instalado
- [ ] Poetry configurado
- [ ] Ambiente virtual ativado
- [ ] Text4GCN importado
- [ ] Testes passando

## 🎯 **Comandos Essenciais**

```bash
# Ativar ambiente
poetry shell

# Executar Python no ambiente
poetry run python

# Instalar dependências
poetry install

# Atualizar dependências
poetry update

# Testar instalação
poetry run python -c "import text4gcn; print('OK!')"
```

## 🚀 **Próximos Passos**

1. **Configurar ambiente** usando uma das soluções acima
2. **Testar instalação** com dados sintéticos
3. **Experimentar** com seus próprios dados
4. **Desenvolver** novas funcionalidades

Com essas soluções, você deve conseguir executar o Text4GCN sem problemas! 🎉
