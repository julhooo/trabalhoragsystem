# RAG Simples com Busca Híbrida

Projeto de prospecção tecnológica sobre **Retrieval-Augmented Generation (RAG)**, utilizando busca híbrida para recuperar informações relevantes e fornecer contexto a um modelo de linguagem (LLM).

A implementação utiliza o framework `smolagents` e o `LiteLLMModel` para integrar o modelo de linguagem ao sistema.

## Tecnologias utilizadas

- **smolagents:** framework utilizado para integrar o modelo de linguagem.
- **LiteLLM:** interface para comunicação com diferentes modelos de linguagem.
- **Sentence Transformers:** geração de embeddings para busca semântica.
- **TF-IDF:** representação numérica dos documentos para busca lexical.
- **Scikit-learn:** implementação da vetorização e do cálculo de similaridade de cosseno.
- **NumPy:** operações numéricas e combinação dos rankings.
- **python-dotenv:** carregamento de variáveis de ambiente.

## Como funciona o projeto?

O projeto implementa um sistema RAG que combina recuperação de informações com geração de texto. Em vez de depender exclusivamente do conhecimento prévio do modelo, o sistema busca documentos relevantes e os fornece como contexto para a resposta.

A aplicação compara dois cenários:

- **Sem RAG:** o modelo recebe apenas a pergunta do usuário.
- **Com RAG:** o sistema recupera documentos relevantes e os envia ao modelo junto com a pergunta.

Essa comparação demonstra como o acesso a uma base documental pode ajudar o modelo a responder perguntas sobre informações específicas.

## Explicação do código

### 1. Configuração do modelo

O código carrega as variáveis de ambiente com `python-dotenv` e inicializa o modelo de linguagem por meio da classe `LiteLLMModel`, do `smolagents`.

Essa camada é responsável por permitir que o sistema envie perguntas ao LLM e receba respostas. A integração depende da configuração do provedor e do modelo escolhido.

### 2. Base de documentos

A variável `DOCUMENTOS` contém cinco textos fictícios sobre a empresa Nimbus Logística.

Os documentos abordam benefícios corporativos, plano de saúde, reembolsos, política de home office e suporte de TI.

Essa lista funciona como a base de conhecimento consultada pelo sistema. Quando o usuário faz uma pergunta, o mecanismo de busca procura os documentos mais relevantes para respondê-la.

### 3. Inicialização dos mecanismos de busca

O código prepara dois mecanismos de recuperação independentes.

**Busca lexical — TF-IDF**

O `TfidfVectorizer` transforma os documentos em vetores numéricos, atribuindo pesos aos termos de acordo com sua importância nos textos.

A similaridade de cosseno é utilizada para comparar o vetor da pergunta com os vetores dos documentos. Assim, o sistema consegue identificar quais textos apresentam maior correspondência lexical com a consulta.

**Busca semântica — Sentence Transformers**

O modelo `all-MiniLM-L6-v2` transforma os documentos em embeddings, representações vetoriais que capturam aspectos do significado dos textos.

Quando uma pergunta é recebida, ela também é convertida em embedding. A similaridade de cosseno permite identificar documentos semanticamente relacionados, mesmo quando a pergunta utiliza palavras diferentes das encontradas nos documentos.

### 4. Recuperação híbrida com RRF

A função `buscar_documentos_hibrido()` combina os resultados das duas estratégias de busca.

Primeiro, calcula as pontuações lexicais e semânticas. Em seguida, cria um ranking para cada mecanismo e utiliza o algoritmo **Reciprocal Rank Fusion (RRF)** para combinar suas posições.

A fórmula utilizada é:

\[
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
\]

Nessa fórmula:

- \(d\) representa um documento.
- \(R\) representa os rankings utilizados.
- \(rank_r(d)\) representa a posição do documento em cada ranking.
- \(k\) é uma constante definida como 60 no código.

O RRF permite combinar rankings sem precisar comparar diretamente as pontuações originais dos dois mecanismos.

Por fim, a função seleciona os dois documentos com as maiores pontuações combinadas e os retorna como contexto para a geração da resposta.

### 5. Geração de respostas com e sem RAG

O código possui duas funções para comparar os resultados.

**`responder_sem_rag()`**

Envia a pergunta diretamente ao modelo de linguagem, sem recuperar ou fornecer os documentos corporativos.

Esse cenário representa o uso convencional de um LLM, no qual a resposta depende das informações disponíveis ao modelo e das instruções recebidas.

**`responder_com_rag()`**

Primeiro, chama `buscar_documentos_hibrido()` para recuperar os documentos mais relevantes.

Em seguida, organiza os documentos em um bloco de contexto e envia esse conteúdo ao modelo junto com instruções restritivas. O modelo é orientado a utilizar exclusivamente as informações recuperadas e a informar quando não encontrar os dados necessários.

A função retorna a resposta gerada e o contexto utilizado, permitindo verificar quais documentos foram fornecidos ao modelo.

### 6. Execução e comparação

Ao executar o script, o programa utiliza a pergunta:

> Qual o valor do nosso vale-refeição e onde ele cai?

A pergunta é enviada primeiro ao modelo sem RAG e, depois, ao sistema com RAG.

No segundo caso, o programa também exibe os documentos recuperados, permitindo observar como a busca influencia o contexto utilizado na geração da resposta.

## Instalação

Instale as dependências necessárias:

```bash
pip install smolagents[litellm] sentence-transformers scikit-learn numpy python-dotenv
```

Configure o .env com uma chave válida para o provedor utilizado na configuração do modelo.

## Execução

Execute o script Python:

```bash
python main.py
```
Na primeira execução, o Sentence Transformers poderá baixar o modelo de embeddings, o que pode aumentar o tempo de inicialização.
