"""
Prospecção Tecnológica: RAG (Retrieval-Augmented Generation) Simples
Uso de Busca Híbrida (Lexical + Semântica) e integração com a API da Groq
através do framework 'smolagents' (LiteLLMModel).

Dependências necessárias:
pip install smolagents[litellm] sentence-transformers scikit-learn numpy python-dotenv
"""

import os
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from smolagents import LiteLLMModel

# =====================================================================
# 1. CONFIGURAÇÃO DA CHAVE E MODELO
# =====================================================================
load_dotenv() 

CHAVE = os.getenv("GROQ_API_KEY")

print("Inicializando o modelo LLM via smolagents...")
modelo_llm = LiteLLMModel(model_id="groq/openai/gpt-oss-120b")

# =====================================================================
# 2. DADOS CORPORATIVOS FICTÍCIOS
# =====================================================================
DOCUMENTOS = [
    "O vale-refeição da Nimbus Logística é de R$ 45,00 por dia útil, creditado no cartão Caju todo dia 01.",
    "O plano de saúde oferecido pela Nimbus Logística aos funcionários é o SulAmérica Exato, com coparticipação de 20% em exames.",
    "Para solicitar reembolso de despesas de viagem, o colaborador deve preencher o formulário no portal RH em até 5 dias úteis após o retorno.",
    "A política de home office da Nimbus Logística permite até 2 dias de trabalho remoto por semana para o setor de TI e Administrativo.",
    "O suporte de TI da Nimbus Logística atende pelo ramal 9999 ou pelo email suporte@nimbus.com.br, com SLA de 4 horas para chamados P1."
]

# =====================================================================
# 3. INICIALIZAÇÃO DOS MOTORES DE BUSCA
# =====================================================================
print("Carregando modelos de busca.")

# Busca Lexical (TF-IDF)
vetorizador_lexical = TfidfVectorizer()
matriz_lexical = vetorizador_lexical.fit_transform(DOCUMENTOS)

# Busca Semântica (SentenceTransformers)
modelo_semantico = SentenceTransformer('all-MiniLM-L6-v2')
matriz_semantica = modelo_semantico.encode(DOCUMENTOS)

# =====================================================================
# 4. LÓGICA DE RECUPERAÇÃO HÍBRIDA (RRF)
# =====================================================================
def buscar_documentos_hibrido(pergunta: str, k: int = 2) -> list[str]:
    """Busca documentos combinando palavras-chave e semântica via RRF."""
    # Score Lexical
    vet_lex = vetorizador_lexical.transform([pergunta])
    scores_lex = cosine_similarity(vet_lex, matriz_lexical).flatten()
    
    # Score Semântico
    vet_sem = modelo_semantico.encode([pergunta])
    scores_sem = cosine_similarity(vet_sem, matriz_semantica).flatten()
    
    # Ranking RRF
    rank_lex = np.argsort(scores_lex)[::-1]
    rank_sem = np.argsort(scores_sem)[::-1]
    
    scores_rrf = np.zeros(len(DOCUMENTOS))
    for i in range(len(DOCUMENTOS)):
        pos_lex = np.where(rank_lex == i)[0][0] + 1
        pos_sem = np.where(rank_sem == i)[0][0] + 1
        scores_rrf[i] = (1 / (60 + pos_lex)) + (1 / (60 + pos_sem))
    
    # Top K documentos
    top_k = np.argsort(scores_rrf)[::-1][:k]
    return [DOCUMENTOS[i] for i in top_k]

# =====================================================================
# 5. GERAÇÃO DE RESPOSTAS COM SMOLAGENTS (COM E SEM RAG)
# =====================================================================
def responder_sem_rag(pergunta: str) -> str:
    """LLM responde com seu conhecimento base, sem receber os documentos."""
    prompt = f"Responda à seguinte pergunta sobre a empresa Nimbus Logística: {pergunta}"
    
    mensagens = [
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    resposta = modelo_llm(mensagens)
    return resposta.content

def responder_com_rag(pergunta: str) -> str:
    """LLM recebe instruções estritas e os documentos recuperados no prompt."""
    documentos_recuperados = buscar_documentos_hibrido(pergunta)
    contexto = "\n\n".join([f"- {doc}" for doc in documentos_recuperados])
    
    prompt_sistema = """Você é um assistente corporativo da empresa Nimbus Logística.
Sua tarefa é responder à pergunta do usuário utilizando EXCLUSIVAMENTE as informações fornecidas no bloco de CONTEXTO abaixo.

REGRAS RÍGIDAS:
1. Baseie sua resposta unicamente nos fatos presentes no CONTEXTO.
2. Não utilize nenhum conhecimento prévio ou externo.
3. Se a informação não estiver explícita no CONTEXTO, responda exatamente: "Não encontrei essa informação nos documentos da empresa."
4. Nunca invente dados, valores ou políticas."""

    prompt_usuario = f"CONTEXTO RECUPERADO:\n{contexto}\n\nPERGUNTA:\n{pergunta}\n\nRESPOSTA BASEADA ESTRITAMENTE NO CONTEXTO:"
    
    # CORREÇÃO: Estruturando o content como lista de blocos de texto
    mensagens = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": prompt_sistema}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt_usuario}]
        }
    ]
    
    resposta = modelo_llm(mensagens)
    return resposta.content, contexto

# =====================================================================
# 6. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DO RAG COM SMOLAGENTS E LITELLM".center(60))
    print("="*60)
    
    pergunta_teste = "Qual o valor do nosso vale-refeição e onde ele cai?"
    print(f"\nPERGUNTA DO USUÁRIO: '{pergunta_teste}'\n")
    
    # --- TESTE 1: SEM RAG ---
    print("-" * 60)
    print("RESPOSTA SEM RAG (LLM 'Puro'):")
    print("-" * 60)
    try:
        resp_sem = responder_sem_rag(pergunta_teste)
        print(resp_sem)
    except Exception as e:
        print(f"Erro na API: {e}")

    # --- TESTE 2: COM RAG ---
    print("\n" + "-" * 60)
    print("RESPOSTA COM RAG (Busca Híbrida + Prompt Restritivo):")
    print("-" * 60)
    try:
        resp_com, contexto_usado = responder_com_rag(pergunta_teste)
        print(f"[Contexto injetado no prompt]:\n{contexto_usado}\n")
        print(f"[Resposta do Assistente]:\n{resp_com}")
    except Exception as e:
        print(f"Erro na API: {e}")