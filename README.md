# Lab 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Lauan Matheus.

---

## Objetivo

Implementar um pipeline de **Retrieval-Augmented Generation (RAG)** de nível de produção para busca semântica em manuais médicos, combinando três técnicas avançadas:

- **HNSW** — índice vetorial hierárquico para busca aproximada ultrarrápida
- **HyDE** — transformação de query coloquial em documento hipotético técnico via LLM
- **Cross-Encoder** — re-ranking de alta precisão por atenção cruzada

---

## Estrutura do Repositório

```
lab09-rag/
├── README.md           # Documentação e análise técnica
├── rag_pipeline.py     # Pipeline completo (Passos 1-4)
└── requirements.txt    # Dependências
```

---

## Como Executar

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sua-chave-aqui"   # Linux/macOS
$env:OPENAI_API_KEY="sua-chave-aqui"     # PowerShell (Windows)

python rag_pipeline.py
```

---

## Descrição do Pipeline

### Passo 1 — Construção do Índice HNSW

22 fragmentos de manuais médicos são convertidos em vetores densos com `text-embedding-3-small` (OpenAI) e indexados em um grafo **HNSW** via FAISS. Os vetores são normalizados para unidade antes da indexação, de modo que o produto interno retornado equivale à **similaridade de cosseno**.

```python
index = faiss.IndexHNSWFlat(dim, M=32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
```

### Passo 2 — HyDE (Hypothetical Document Embeddings)

Quando o paciente digita `"dor de cabeça latejante e luz incomodando muito"`, o LLM (`gpt-4o-mini`) é instruído a **alucinar** um trecho técnico de manual que descreveria esse quadro clínico — por exemplo:

> *"Cefaleia pulsátil unilateral acompanhada de fotofobia intensa e fonofobia, compatível com episódio de enxaqueca sem aura…"*

Esse documento falso é vetorizado e serve como **âncora geométrica** no espaço dos embeddings técnicos, onde o jargão médico já existe — eliminando o *vocabulary mismatch* entre linguagem coloquial e terminologia clínica.

### Passo 3 — Busca Rápida no HNSW (Bi-Encoder, Top-10)

O vetor HyDE é comparado com todos os vetores indexados via busca aproximada de vizinhos mais próximos no grafo HNSW. Retorna os **Top-10** fragmentos mais similares (funil largo).

### Passo 4 — Re-ranking com Cross-Encoder (Top-3)

Os 10 candidatos são re-ranqueados pelo modelo `cross-encoder/ms-marco-MiniLM-L-6-v2`, que processa o par `[CLS] query_original [SEP] documento` com **atenção cruzada bidirecional** — capturando interações sutis entre os termos. Os **Top-3** resultantes são os documentos injetados no contexto do LLM gerador.

---

## Análise: HNSW vs KNN — Impacto dos Hiperparâmetros na RAM

### Custo de memória do KNN exato

Na busca exata K-Nearest Neighbors, é necessário armazenar todos os vetores e calcular a distância de cada query contra o corpus inteiro em tempo de execução:

```
RAM_KNN = N × dim × 4 bytes
```

Para N = 1 000 000 vetores com dim = 1536 (text-embedding-3-small):
```
RAM_KNN ≈ 1M × 1536 × 4 = ~5,9 GB
```

O tempo de busca é **O(N × dim)** por query — inviável em produção com grandes corpora.

### Custo de memória do HNSW

O HNSW adiciona uma estrutura de grafo em camadas sobre os vetores brutos. O custo extra depende de dois hiperparâmetros:

**M (número de ligações bidirecionais por nó)**

- Cada nó armazena até `2 × M` vizinhos (IDs inteiros de 4 bytes).
- Custo extra do grafo: `N × 2M × 4 bytes`
- M maior → mais conexões → melhor recall → **mais RAM**
- M típico: 16–64

**ef_construction (tamanho da lista dinâmica na construção)**

- Controla quantos candidatos são avaliados ao inserir cada nó no grafo durante o *build*.
- Afeta apenas a **RAM temporária durante a indexação** (não o índice final em disco/memória).
- ef_construction maior → índice de maior qualidade (recall) → build mais lento
- Não altera o footprint do índice em produção.

### Comparação para N = 1M, dim = 1536, M = 32

| Componente         | Fórmula                    | Tamanho     |
|--------------------|----------------------------|-------------|
| Vetores brutos     | N × dim × 4                | ~5,9 GB     |
| Grafo HNSW (M=32)  | N × 2M × 4                 | ~256 MB     |
| **Total HNSW**     |                            | **~6,2 GB** |
| **KNN exato**      | apenas vetores brutos       | **~5,9 GB** |

O HNSW consome **ligeiramente mais RAM** que KNN puro (pela estrutura do grafo), mas reduz o tempo de busca de **O(N)** para **O(log N)**, tornando a consulta viável em produção. A economia real é de **tempo** e **CPU** — não de memória bruta.

**Resumo prático:**

- Aumentar `M`: melhora recall, aumenta RAM do grafo e tempo de build.
- Aumentar `ef_construction`: melhora qualidade do índice, sem custo em produção.
- HNSW escala para bilhões de vetores onde KNN exato se torna computacionalmente inviável.

---

## Dependências

```
openai>=1.30.0
faiss-cpu>=1.7.4
sentence-transformers>=2.7.0
numpy>=1.24.0
```
