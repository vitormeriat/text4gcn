# PyTorch Device Management in Text4GCN

Este documento explica como o Text4GCN implementa a melhor prática de mercado para gerenciamento de dispositivos PyTorch (CPU vs GPU).

## 🎯 **Abordagem Implementada**

### **Extras do Poetry (Melhor Prática de Mercado)**

A implementação segue a **melhor prática de mercado** usada por pacotes como:
- PyTorch
- Transformers (Hugging Face)
- Lightning
- Scikit-learn

### **Estrutura de Extras**

```toml
[tool.poetry.extras]
# PyTorch CPU version (default)
cpu = ["torch>=2.1.2,<2.2.0"]
# PyTorch GPU version (CUDA)
gpu = ["torch>=2.1.2,<2.2.0+cu121"]
# All extras
all = ["torch>=2.1.2,<2.2.0", "torch>=2.1.2,<2.2.0+cu121"]
```

## 📦 **Opções de Instalação**

### **1. CPU Version (Padrão)**
```bash
pip install text4gcn[cpu]
```

### **2. GPU Version (CUDA)**
```bash
pip install text4gcn[gpu]
```

### **3. Instalação Manual**
```bash
# Instalar sem PyTorch
pip install text4gcn

# Depois instalar PyTorch baseado no sistema
# Para CPU:
pip install torch

# Para GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### **4. Desenvolvimento**
```bash
# CPU
poetry install --extras "cpu"

# GPU
poetry install --extras "gpu"
```

## 🔧 **Device Manager**

### **Funcionalidades Implementadas**

1. **Auto-detecção de dispositivo**
2. **Gerenciamento de dispositivos**
3. **Informações detalhadas do sistema**
4. **Recomendações de instalação**
5. **Tratamento de erros**

### **Uso Básico**

```python
import text4gcn

# Auto-detectar dispositivo
device_manager = text4gcn.utils.get_device_manager()
print(f"Using device: {device_manager.device}")

# Informações do dispositivo
info = device_manager.get_device_info()
print(f"Device info: {info}")

# Verificar instalação do PyTorch
pytorch_info = text4gcn.utils.check_pytorch_installation()
print(f"PyTorch available: {pytorch_info['torch_available']}")
```

### **Uso Avançado**

```python
# Forçar CPU
cpu_manager = text4gcn.utils.get_device_manager("cpu")

# Forçar GPU (se disponível)
gpu_manager = text4gcn.utils.get_device_manager("cuda")

# Verificar recomendações
recommendation = text4gcn.utils.recommend_installation()
print(recommendation)
```

## 🤖 **Integração com Modelos**

### **GNN com Device Management**

```python
# Auto-detecção
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    device=None  # Auto-detect
)

# Forçar CPU
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    device="cpu"  # Force CPU
)

# Forçar GPU
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    device="cuda"  # Force GPU
)
```

## 📊 **Compatibilidade de Versões**

| Text4GCN | PyTorch CPU | PyTorch GPU (CUDA) |
|----------|-------------|-------------------|
| 2.0.0+   | 2.1.2+      | 2.1.2+ (CUDA 12.1) |

## 🧪 **Testes Implementados**

### **Cobertura de Testes**

- ✅ Testes unitários para DeviceManager
- ✅ Testes de integração
- ✅ Testes de performance
- ✅ Testes de error handling
- ✅ Testes com PyTorch real (se disponível)

### **Executar Testes**

```bash
# Todos os testes
pytest tests/test_device.py

# Testes específicos
pytest tests/test_device.py::TestDeviceManager

# Com cobertura
pytest tests/test_device.py --cov=text4gcn.utils.device
```

## 🔍 **Detecção de Sistema**

### **Informações Disponíveis**

```python
# Verificar instalação
info = text4gcn.utils.check_pytorch_installation()
print(info)
# {
#     'torch_available': True,
#     'torch_version': '2.1.2',
#     'cuda_available': True,
#     'cuda_version': '12.1'
# }

# Informações do dispositivo
device_info = device_manager.get_device_info()
print(device_info)
# {
#     'device': 'cuda',
#     'type': 'cuda',
#     'is_cuda': True,
#     'is_cpu': False,
#     'cuda_version': '12.1',
#     'gpu_name': 'NVIDIA GeForce RTX 4090',
#     'gpu_memory': 24576000000,
#     'gpu_memory_gb': 24.6
# }
```

## 🚀 **Vantagens da Implementação**

### **1. Flexibilidade**
- Suporte a CPU e GPU
- Auto-detecção inteligente
- Instalação manual possível

### **2. Compatibilidade**
- Segue padrões da indústria
- Compatível com Poetry
- Suporte a diferentes versões do PyTorch

### **3. Usabilidade**
- API simples e intuitiva
- Mensagens de erro claras
- Recomendações automáticas

### **4. Manutenibilidade**
- Código bem testado
- Documentação completa
- Exemplos práticos

## 📝 **Exemplos Práticos**

### **Exemplo 1: Verificação de Sistema**

```python
import text4gcn

# Verificar se PyTorch está instalado
info = text4gcn.utils.check_pytorch_installation()
if not info['torch_available']:
    print("PyTorch não está instalado!")
    recommendation = text4gcn.utils.recommend_installation()
    print(f"Instale com: {recommendation}")
else:
    print(f"PyTorch {info['torch_version']} disponível")
    if info['cuda_available']:
        print(f"CUDA {info['cuda_version']} disponível")
```

### **Exemplo 2: Treinamento com Device Management**

```python
import text4gcn

# Configurar dispositivo
device_manager = text4gcn.utils.get_device_manager()
print(f"Usando dispositivo: {device_manager.device}")

# Treinar modelo
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    device=device_manager.device
)
model.fit()
```

## 🎯 **Melhores Práticas**

### **1. Para Desenvolvimento**
```bash
# Instalar versão de desenvolvimento
poetry install --extras "gpu"  # se tiver GPU
poetry install --extras "cpu"  # se só tiver CPU
```

### **2. Para Produção**
```bash
# Instalar versão específica
pip install text4gcn[gpu]  # para servidores com GPU
pip install text4gcn[cpu]  # para servidores CPU-only
```

### **3. Para CI/CD**
```bash
# Usar versão CPU para testes
pip install text4gcn[cpu]
```

## 🚀 **Conclusão**

A implementação do gerenciamento de dispositivos PyTorch no Text4GCN segue as **melhores práticas de mercado**, oferecendo:

- ✅ **Flexibilidade**: Suporte a CPU e GPU
- ✅ **Simplicidade**: API intuitiva e auto-detecção
- ✅ **Robustez**: Tratamento de erros e validações
- ✅ **Compatibilidade**: Segue padrões da indústria
- ✅ **Manutenibilidade**: Código bem testado e documentado

O pacote agora está pronto para uso em diferentes ambientes, desde desenvolvimento local até produção em servidores com GPU! 🎉
