# gpt-from-scratch

Uma implementação mínima de GPT em PyTorch, treinada em Shakespeare

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Licença MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Abrir no Colab](https://img.shields.io/badge/Open%20in-Colab-f9ab00)](https://colab.research.google.com/)

## O que eu construí

Este projeto é uma implementação do zero de um pequeno language model baseado em Transformer e inspirado no GPT. Ele é treinado no corpus Shakespeare de 1.1MB usando tokenização character-level, então cada token de entrada é um único caractere. O objetivo é entender a arquitetura GPT implementando manualmente cada componente principal em vez de tratar o modelo como uma caixa-preta.

## Arquitetura

```txt
Tokens de entrada
    |
    v
Token Embedding + Positional Embedding
    |
    v
N x Transformer Block
    |
    +--> LayerNorm -> Multi-Head Attention -> Soma Residual
    |
    +--> LayerNorm -> FeedForward -> Soma Residual
    |
    v
LayerNorm
    |
    v
Cabeça Linear
    |
    v
Logits
```

| Parâmetro | Valor |
|---|---:|
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| block_size | 256 |
| batch_size | 64 |
| max_iters | 5000 |
| learning_rate | 3e-4 |
| dropout | 0.2 |

## Resultados

A validação de treino rápido, usando 500 iterações e um modelo menor amigável para CPU, chegou a uma validation loss de **2.3975**. Um treino completo no Colab com a configuração completa deve chegar aproximadamente a **1.5-1.7** de validation loss.

![Curva de loss](assets/loss_curve.png)

Exemplo gerado pelo treino rápido de 500 iterações:

```txt
ININORCIRRPDS:
I ouro y the anisie fof my isthacas t Wa hefis ar acha ar
Be oke ld:
Thof f ve thisilit de din ond pe pr;
Milthn heaiche t ourd mon yo d owis ion fatel blem tha t omy ch
```

O texto incoerente é esperado para um smoke test muito curto em CPU. Com o treino completo no Colab, o modelo deve produzir texto com uma estrutura mais reconhecível de Shakespeare, incluindo quebras de linha, nomes e padrões de diálogo no estilo de peças teatrais.

## Componentes Principais

**Causal Self-Attention:** Queries, keys e values são projetados manualmente com camadas lineares. Uma máscara causal triangular inferior impede que cada token preste atenção em posições futuras.

**Positional Embedding:** O modelo usa positional embeddings aprendidas em vez de codificações sinusoidais. Isso permite que o modelo aprenda diretamente dos dados como a ordem dos tokens importa.

**Weight Tying:** A matriz de token embedding e a cabeça do language model compartilham os mesmos pesos. Isso reduz o número de parâmetros e segue um padrão comum em modelos no estilo GPT.

**Pre-norm:** LayerNorm é aplicado antes de cada sub-layer de atenção e feed-forward. Esse layout no estilo GPT-2 normalmente torna o treinamento de Transformers mais estável.

## Estrutura do Projeto

```txt
gpt-from-scratch/
├── assets/           # gráficos de treino e amostras geradas
├── checkpoints/      # pesos salvos do modelo (ignorado pelo git)
├── data/
│   └── tiny_shakespeare.txt
├── model/
│   ├── attention.py  # Head, MultiHeadAttention, FeedForward, Block
│   ├── gpt.py        # GPTLanguageModel, build_model
│   └── __init__.py
├── notebooks/        # notebooks exploratórios e de visualização
├── utils/
│   └── tokenizer.py  # CharacterTokenizer, ShakespeareDataset
├── config.py         # todos os hiperparâmetros em um só lugar
├── train.py          # loop de treino com AdamW e checkpointing
└── generate.py       # geração de texto com argumentos de CLI
```

## Como Rodar

### Local

```bash
git clone https://github.com/artur-source/gpt-from-scratch
cd gpt-from-scratch
pip install -r requirements.txt
python train.py
python generate.py --prompt "To be or not to be" --max_tokens 300 --temperature 0.8 --top_k 40
```

### Google Colab

Abra o projeto usando o badge do Colab no topo deste README. No Colab, use uma GPU T4 selecionando **Runtime -> Change runtime type -> T4 GPU**. Um treino completo com a configuração completa deve levar cerca de **15-20 minutos**.

## Referências

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy

Construído por Artur como projeto de portfólio - Análise e Desenvolvimento de Sistemas, UniPiaget
