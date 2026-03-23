# Transformer From Scratch - Lab 5

##  Descrição

Este projeto implementa um Transformer completo (Encoder + Decoder) utilizando NumPy e adiciona um pipeline de treinamento com PyTorch.

O objetivo é demonstrar tanto a arquitetura interna quanto o processo de treinamento de um modelo Transformer.

---

##  Arquitetura

O modelo é composto por:

###  Encoder
- Self-Attention
- Add & Norm
- Feed Forward Network

###  Decoder
- Masked Self-Attention
- Cross-Attention
- Camada Linear + Softmax

---

##  Pipeline de Treinamento

O treinamento segue o fluxo:

1. Carregamento de dataset (Hugging Face)
2. Tokenização de texto
3. Forward pass
4. Cálculo da loss (CrossEntropy)
5. Backpropagation
6. Atualização dos pesos (Adam)

---

##  Dataset

Utilizado:
- `bentrevett/multi30k` (subconjunto)

---
## Nota sobre IA
Partes do código foram geradas/complementadas com auxílio de IA,
sendo totalmente revisadas e compreendidas."""
"""
---

##  Como Executar

```bash
pip install -r requirements.txt
python train.py
