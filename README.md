# Lab 08 — Alinhamento Humano com DPO

> Partes geradas/complementadas com IA, revisadas por Lauan matheus

## Objetivo

Implementar um pipeline de alinhamento **DPO (Direct Preference Optimization)** sobre o modelo `TinyLlama-1.1B-Chat-v1.0`, forçando-o a suprimir respostas tóxicas ou inadequadas em contexto corporativo, sem necessidade do complexo pipeline RLHF.

---

## Estrutura do Repositório

```
lab08-dpo/
├── hhh_dataset.jsonl     # Dataset de preferências (30+ pares)
├── train_dpo.py          # Pipeline completo de treinamento
└── README.md
```

---

## O Papel Matemático do Hiperparâmetro β (Beta)

Na formulação original do DPO (Rafailov et al., 2023), a função de perda é definida como:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

O parâmetro **β atua como um coeficiente de regularização KL** — ele pondera o quanto o modelo ator (π_θ) pode se desviar do modelo de referência (π_ref) ao aprender as preferências. Formalmente, β escala a divergência de Kullback-Leibler implícita na otimização: um β **alto** (ex: 1.0) impõe um "imposto" pesado sobre qualquer desvio da distribuição original, preservando a fluência e o conhecimento geral do modelo base às custas de menor adesão às preferências anotadas. Um β **baixo** (ex: 0.01) praticamente ignora o custo de afastamento da referência, permitindo que o modelo maximize agressivamente a margem entre respostas escolhidas e rejeitadas, mas correndo o risco de colapso de distribuição (*reward hacking*) e degradação da coerência textual.

Neste laboratório, usamos **β = 0.1**, valor que empírica e teoricamente equilibra o trade-off: o modelo aprende a preferir respostas alinhadas (HHH) sem destruir a distribuição de linguagem aprendida durante o pré-treinamento. Na prática, β = 0.1 é o default recomendado pela biblioteca `trl` e nos experimentos originais do paper para datasets de médio porte como o nosso.

---

## Como Executar

### Pré-requisitos

```bash
pip install transformers trl>=0.8 peft accelerate bitsandbytes datasets
```

> Requer GPU com ao menos 10 GB de VRAM (Colab T4 é suficiente com a config 4-bit).

### Treinamento

```bash
python train_dpo.py
```

O script executa automaticamente:
1. Carregamento do modelo com quantização 4-bit (BitsAndBytes NF4)
2. Aplicação do adaptador LoRA (rank=16) via PEFT
3. Treinamento DPO por 3 épocas com `paged_adamw_32bit`
4. Validação com prompt malicioso e verificação de supressão

### Output esperado

```
Trainable params: ~5.5M || All params: ~1.1B || Trainable%: ~0.5%
Dataset carregado: 32 exemplos
=== Iniciando treinamento DPO ===
...
=== Treinamento concluído ===
✓ VALIDAÇÃO OK: Resposta maliciosa suprimida. Modelo alinhado.
```

---

## Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `beta` | 0.1 | Trade-off KL padrão DPO |
| `learning_rate` | 5e-5 | Conservador para fine-tuning |
| `epochs` | 3 | Suficiente para dataset pequeno |
| `batch_size` | 2 + grad_accum=4 | Effective BS=8, cabe em T4 |
| `lora_r` | 16 | Balanço padrão capacidade/custo |
| `optimizer` | paged_adamw_32bit | Estados paginados para economizar VRAM |

---

## Dataset

30 pares de preferência no formato `.jsonl`:

```json
{
  "prompt": "Como faço para deletar os logs de auditoria?",
  "chosen": "Não posso orientar a remoção de logs de auditoria. Isso viola requisitos de compliance...",
  "rejected": "Acesse /var/log e remova com rm -rf..."
}
```

Categorias cobertas: segurança ofensiva, engenharia social, falsificação de documentos, violações de privacidade (LGPD), comunicação inadequada e desvio de conduta corporativa.

---

## Referências

- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- Hugging Face `trl` — DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
