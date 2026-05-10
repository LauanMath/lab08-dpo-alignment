#!/usr/bin/env python3
"""
Lab 09 — Arquitetura RAG Avançada: HNSW + HyDE + Cross-Encoder
Assistente de busca semântica em manuais médicos.
"""

import os
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import CrossEncoder

# ── Corpus: 22 fragmentos de manuais médicos ─────────────────────────────────
MEDICAL_DOCS = [
    "Cefaleia pulsátil (enxaqueca) é caracterizada por dor latejante unilateral, fotofobia, fonofobia e náusea. Pode durar de 4 a 72 horas sem tratamento.",
    "Fotofobia é a hipersensibilidade anormal à luz, comumente associada a enxaqueca, meningite e uveíte. O paciente refere desconforto intenso em ambientes iluminados.",
    "Fonofobia refere-se à hipersensibilidade ao som. Sintoma clássico da crise migranosa, frequentemente acompanhada de fotofobia e náusea.",
    "Náusea e vômito durante episódios de cefaleia indicam comprometimento do sistema nervoso autônomo, típico de enxaqueca com ou sem aura.",
    "Aura migranosa: sintomas neurológicos transitórios que precedem a cefaleia — escotomas cintilantes, parestesias unilaterais e disartria temporária.",
    "Meningite bacteriana apresenta tríade clássica: febre alta, rigidez de nuca (sinal de Kernig positivo) e cefaleia intensa. Emergência médica.",
    "Hipertensão intracraniana: cefaleia progressiva que piora ao decúbito, vômitos em jato e papiledema ao exame de fundo de olho.",
    "Cefaleia tensional: bilateral, em pressão ou aperto, intensidade leve a moderada, sem náusea significativa. Associada a estresse e tensão muscular cervical.",
    "Neuralgia do trigêmeo: dor facial unilateral súbita tipo choque elétrico, duração de segundos, desencadeada por toque, mastigação ou fala.",
    "Crise hipertensiva: PA sistólica > 180 mmHg precipita cefaleia severa, geralmente occipital matinal. Pode evoluir para encefalopatia hipertensiva.",
    "AVC isquêmico: cefaleia súbita intensa, hemiplegia, afasia e desvio conjugado do olhar. Janela terapêutica de 4,5 h para trombólise IV.",
    "Hemorragia subaracnóidea por aneurisma roto: cefaleia em trovoada (thunderclap headache) de início abrupto, com rigidez de nuca e fotofobia.",
    "Glaucoma agudo de ângulo fechado: dor ocular intensa com cefaleia ipsilateral, visão turva com halos coloridos e midríase fixa. Emergência oftalmológica.",
    "Sinusite aguda: pressão e cefaleia frontal ou maxilar que piora ao inclinar a cabeça, secreção nasal purulenta e febre baixa.",
    "Cefaleia em salvas (cluster headache): unilateral, orbitária, excruciante, duração de 15-180 min, com lacrimejamento e rinorreia ipsilateral.",
    "Hipoglicemia: glicemia < 70 mg/dL causa cefaleia, sudorese, tremores, confusão mental e visão turva. Tratamento: carboidratos de absorção rápida.",
    "Trombose venosa cerebral: cefaleia progressiva, convulsões focais e déficits neurológicos variáveis. Risco elevado no puerpério e uso de anticoncepcionais orais.",
    "Miastenia gravis: ptose palpebral, diplopia, disfagia e fraqueza muscular flutuante ao longo do dia. SFEMG é o exame de maior sensibilidade diagnóstica.",
    "Esclerose múltipla: neurite óptica (dor ao movimento ocular com perda visual), parestesias, ataxia e fenômeno de Lhermitte (choque ao flexionar a cervical).",
    "Hipertireoidismo: taquicardia, tremor fino, perda de peso, intolerância ao calor, exoftalmia, diarreia e hiperreflexia. TSH suprimido no diagnóstico.",
    "Hipotireoidismo: bradicardia, ganho de peso, intolerância ao frio, constipação, pele seca e mixedema. TSH elevado com T4 livre reduzido confirma.",
    "Diabetes mellitus tipo 2: hiperglicemia crônica com polidipsia, poliúria, polifagia e perda de peso involuntária. HbA1c >= 6,5% confirma o diagnóstico.",
]

# ── Hiperparâmetros HNSW ──────────────────────────────────────────────────────
HNSW_M          = 32   # ligações bidirecionais por nó (controla memória e recall)
EF_CONSTRUCTION = 200  # lista dinâmica durante a construção (qualidade do grafo)

# ── Modelos ───────────────────────────────────────────────────────────────────
EMBED_MODEL        = "text-embedding-3-small"
LLM_MODEL          = "gpt-4o-mini"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 1 — Embeddings e Indexação HNSW
# ─────────────────────────────────────────────────────────────────────────────

def embed(texts: list[str], client: OpenAI) -> np.ndarray:
    """Gera vetores normalizados para unidade (L2=1) via OpenAI Embeddings."""
    resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
    vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)   # normaliza in-place → produto interno == cosseno
    return vecs


def build_hnsw_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Constrói um índice FAISS-HNSW com similaridade de cosseno.
    Para vetores unitários, produto interno == similaridade de cosseno.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(embeddings)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 2 — HyDE: Geração do Documento Hipotético
# ─────────────────────────────────────────────────────────────────────────────

def generate_hypothetical_document(query: str, client: OpenAI) -> str:
    """
    Pede ao LLM que alucine um trecho técnico de manual médico para a query
    coloquial do paciente (Hypothetical Document Embeddings — HyDE).
    """
    prompt = (
        "Você é um especialista em semiologia médica. "
        "Dado o relato coloquial de um paciente, escreva o fragmento que um manual clínico "
        "usaria para descrever esse quadro, com terminologia médica precisa. "
        "Produza apenas o fragmento — não responda ao paciente.\n\n"
        f'Relato do paciente: "{query}"\n\n'
        "Fragmento do manual médico:"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 3 — Busca Rápida no HNSW (Bi-Encoder, funil largo)
# ─────────────────────────────────────────────────────────────────────────────

def hnsw_search(
    query_vec: np.ndarray,
    index: faiss.IndexHNSWFlat,
    k: int = 10,
    ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Busca os k vizinhos mais próximos (maior cosseno) no índice HNSW."""
    index.hnsw.efSearch = ef_search
    q = query_vec.reshape(1, -1).copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return scores[0], idxs[0]


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 4 — Re-ranking com Cross-Encoder (filtro fino)
# ─────────────────────────────────────────────────────────────────────────────

def cross_encoder_rerank(
    query: str,
    candidates: list[str],
    top_k: int = 3,
) -> list[tuple[float, str]]:
    """
    Re-ranqueia candidatos com Cross-Encoder de atenção cruzada.
    Formato de entrada: [CLS] query [SEP] documento.
    """
    ce = CrossEncoder(CROSS_ENCODER_NAME)
    pairs = [[query, doc] for doc in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores.tolist(), candidates), reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sep = "─" * 65

    # ── PASSO 1 ──────────────────────────────────────────────────────────────
    print(sep)
    print("PASSO 1 — Construção do Índice HNSW")
    print(sep)
    print(f"  Corpus  : {len(MEDICAL_DOCS)} fragmentos de manual médico")
    print(f"  Modelo  : {EMBED_MODEL}")
    print("  Gerando embeddings …")

    doc_vecs = embed(MEDICAL_DOCS, client)
    print(f"  Shape dos embeddings : {doc_vecs.shape}")

    index = build_hnsw_index(doc_vecs)
    print(
        f"  Índice HNSW pronto   | ntotal={index.ntotal}"
        f"  M={HNSW_M}  ef_construction={EF_CONSTRUCTION}"
    )

    # ── PASSO 2 ──────────────────────────────────────────────────────────────
    query = "dor de cabeça latejante e luz incomodando muito"

    print(f"\n{sep}")
    print("PASSO 2 — HyDE: Geração do Documento Hipotético")
    print(sep)
    print(f"  Query coloquial : {query}")
    print("  Gerando documento hipotético via LLM …")

    hyp_doc = generate_hypothetical_document(query, client)
    print(f"\n  Documento hipotético gerado:\n    {hyp_doc}")

    hyde_vec = embed([hyp_doc], client)[0]
    print(f"\n  Vetor HyDE gerado | dim={hyde_vec.shape[0]}")

    # ── PASSO 3 ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PASSO 3 — Recuperação Rápida via HNSW (Top-10 | Funil Largo)")
    print(sep)

    scores, idxs = hnsw_search(hyde_vec, index, k=10)
    top10: list[str] = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), 1):
        doc = MEDICAL_DOCS[idx]
        top10.append(doc)
        preview = doc[:110] + ("…" if len(doc) > 110 else "")
        print(f"\n  [{rank:02d}] cosseno={score:.4f}")
        print(f"       {preview}")

    # ── PASSO 4 ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PASSO 4 — Re-ranking com Cross-Encoder (Top-3 Finais)")
    print(sep)
    print(f"  Modelo : {CROSS_ENCODER_NAME}")
    print("  Aplicando atenção cruzada nos 10 candidatos …")

    top3 = cross_encoder_rerank(query, top10, top_k=3)

    print("\n  Documentos finais — seriam injetados no contexto do LLM gerador:\n")
    for rank, (score, doc) in enumerate(top3, 1):
        print(f"  [{rank}] cross-encoder score = {score:.4f}")
        print(f"      {doc}\n")


if __name__ == "__main__":
    main()
