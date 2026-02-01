# Text4GCN Synthetic Data Testing

Este documento explica como usar dados sintéticos para testar o Text4GCN, seguindo o mesmo formato dos datasets de benchmark (R8, R52, etc.).

## 🎯 **Objetivo**

Criar dados sintéticos no formato correto para testar o `TextPipeline` e outras funcionalidades do Text4GCN sem depender dos datasets reais de benchmark.

## 📁 **Estrutura de Arquivos**

### **Arquivos de Entrada (Formato Text4GCN)**
```
synthetic_data/
├── SyntheticR8.txt      # Documentos (um por linha)
├── SyntheticR8.meta     # Metadados (id, split, label)
└── ...
```

### **Arquivos de Saída (Após TextPipeline)**
```
synthetic_data/
├── SyntheticR8.cleaned/           # Dados limpos
│   └── SyntheticR8.txt
├── SyntheticR8.shuffled/          # Dados embaralhados
│   ├── SyntheticR8.txt
│   ├── SyntheticR8.train
│   ├── SyntheticR8.test
│   ├── SyntheticR8.meta
│   ├── SyntheticR8.vocab
│   └── SyntheticR8.word_vectors
├── SyntheticR8.node_features/     # Features dos nós
│   ├── ind.SyntheticR8.x
│   ├── ind.SyntheticR8.y
│   ├── ind.SyntheticR8.tx
│   ├── ind.SyntheticR8.ty
│   ├── ind.SyntheticR8.allx
│   └── ind.SyntheticR8.ally
└── log/                          # Logs de processamento
    └── SyntheticR8_dataset.txt
```

## 🛠️ **Ferramentas Implementadas**

### **1. Gerador de Dados Sintéticos**
- `synthetic_data_generator.py` - Gerador completo com múltiplas categorias
- `simple_synthetic_test.py` - Teste simples e rápido

### **2. Scripts de Teste**
- `test_textpipeline_synthetic.py` - Teste completo do TextPipeline
- `run_synthetic_test.py` - Script de execução rápida

### **3. Notebook Jupyter**
- `synthetic_data_demo.ipynb` - Demonstração interativa

## 🚀 **Uso Rápido**

### **1. Gerar Dados Sintéticos**

```bash
# Dados simples (100 documentos)
python examples/simple_synthetic_test.py

# Dados completos (1000 documentos)
python examples/synthetic_data_generator.py --dataset-name SyntheticR8 --num-documents 1000

# Múltiplos datasets
python examples/synthetic_data_generator.py --multiple
```

### **2. Testar TextPipeline**

```python
from text4gcn.preprocess import TextPipeline

# Criar pipeline
pipe = TextPipeline(
    dataset_name="SyntheticR8",
    rare_count=5,
    dataset_path="synthetic_data",
    language="english"
)

# Executar processamento
pipe.execute()
```

### **3. Usar com Builders**

```python
from text4gcn.builder import Frequency

# Construir matriz de adjacência
builder = Frequency()
adjacency = builder.tfidf(
    dataset_name="SyntheticR8",
    dataset_path="synthetic_data"
)
```

### **4. Treinar Modelo**

```python
from text4gcn.models import GNN, Layer, Builder

model = GNN(
    dataset="SyntheticR8",
    path="synthetic_data",
    log_dir="logs",
    builder=Builder.Tfidf,
    layer=Layer.GCN,
    epochs=50
)
model.fit()
```

## 📊 **Categorias Sintéticas**

### **Tecnologia**
- Palavras-chave: computer, software, programming, algorithm, data, network
- Templates: "The {keyword} system provides advanced {feature} capabilities"

### **Ciência**
- Palavras-chave: research, experiment, hypothesis, analysis, discovery, theory
- Templates: "The {keyword} study reveals important {feature} findings"

### **Negócios**
- Palavras-chave: company, market, revenue, profit, investment, strategy
- Templates: "The {keyword} strategy increases {feature} performance"

### **Esportes**
- Palavras-chave: team, player, game, match, championship, training
- Templates: "The {keyword} team achieves {feature} success"

### **Saúde**
- Palavras-chave: medical, treatment, patient, health, medicine, therapy
- Templates: "The {keyword} treatment improves {feature} outcomes"

## 🔧 **Configurações Avançadas**

### **Gerador de Dados**

```python
from synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator("output_dir")

# Dataset pequeno para testes rápidos
generator.generate_dataset("SyntheticSmall", 100, 0.7)

# Dataset grande para testes de performance
generator.generate_dataset("SyntheticLarge", 5000, 0.8)

# Múltiplos datasets
generator.generate_multiple_datasets()
```

### **Parâmetros do TextPipeline**

```python
pipe = TextPipeline(
    dataset_name="SyntheticR8",
    rare_count=5,           # Palavras raras (freq < 5)
    dataset_path="synthetic_data",
    language="english"       # Idioma para stop words
)
```

## 📈 **Casos de Uso**

### **1. Desenvolvimento e Testes**
- Testar funcionalidades sem dados reais
- Validar pipeline de processamento
- Debug de problemas específicos

### **2. Performance Testing**
- Testar com datasets grandes
- Medir tempo de processamento
- Otimizar configurações

### **3. Demonstrações**
- Mostrar funcionalidades do Text4GCN
- Treinar usuários
- Documentar uso

### **4. CI/CD**
- Testes automatizados
- Validação de builds
- Verificação de regressões

## 🧪 **Exemplos de Teste**

### **Teste Básico**
```bash
cd examples
python simple_synthetic_test.py
```

### **Teste Completo**
```bash
cd examples
python test_textpipeline_synthetic.py
```

### **Notebook Interativo**
```bash
cd examples
jupyter notebook synthetic_data_demo.ipynb
```

## 📝 **Formato dos Dados**

### **Arquivo .txt**
```
computer software programming algorithm data network system digital internet cloud technology innovation development the and of to in is it that for with as was be on have
research experiment hypothesis analysis discovery theory methodology observation conclusion study scientific investigation the and of to in is it that for with as was be on have
company market revenue profit investment strategy growth management finance economy sales customer business development the and of to in is it that for with as was be on have
```

### **Arquivo .meta**
```
0	train	technology
1	train	science
2	train	business
3	test	technology
4	test	science
```

## 🔍 **Verificação de Saída**

### **Arquivos Esperados**
- ✅ `{dataset}.cleaned/{dataset}.txt` - Dados limpos
- ✅ `{dataset}.shuffled/{dataset}.txt` - Dados embaralhados
- ✅ `{dataset}.shuffled/{dataset}.train` - Índices de treino
- ✅ `{dataset}.shuffled/{dataset}.test` - Índices de teste
- ✅ `{dataset}.shuffled/{dataset}.meta` - Metadados embaralhados
- ✅ `{dataset}.shuffled/{dataset}.vocab` - Vocabulário
- ✅ `{dataset}.shuffled/{dataset}.word_vectors` - Vetores de palavras
- ✅ `{dataset}.node_features/` - Features dos nós
- ✅ `log/{dataset}_dataset.txt` - Logs de processamento

### **Verificação Programática**
```python
from pathlib import Path

data_path = Path("synthetic_data")
dataset_name = "SyntheticR8"

# Verificar arquivos de saída
output_dirs = [
    f"{dataset_name}.cleaned",
    f"{dataset_name}.shuffled",
    f"{dataset_name}.node_features"
]

for dir_name in output_dirs:
    dir_path = data_path / dir_name
    if dir_path.exists():
        files = list(dir_path.glob("*"))
        print(f"✅ {dir_name}: {len(files)} files")
    else:
        print(f"❌ {dir_name}: missing")
```

## 🚀 **Vantagens dos Dados Sintéticos**

### **1. Controle Total**
- Tamanho dos datasets
- Distribuição das categorias
- Conteúdo dos documentos
- Estrutura dos metadados

### **2. Reproducibilidade**
- Mesmos dados a cada execução
- Resultados consistentes
- Debug facilitado

### **3. Flexibilidade**
- Diferentes tamanhos
- Múltiplas categorias
- Configurações personalizadas

### **4. Performance**
- Testes rápidos
- Validação eficiente
- Desenvolvimento ágil

## 📚 **Próximos Passos**

1. **Experimentar com diferentes tamanhos de dataset**
2. **Testar com diferentes configurações do TextPipeline**
3. **Validar com datasets reais de benchmark**
4. **Otimizar parâmetros para melhor performance**
5. **Integrar com testes automatizados**

Os dados sintéticos fornecem uma base sólida para testar e desenvolver com Text4GCN! 🎉
