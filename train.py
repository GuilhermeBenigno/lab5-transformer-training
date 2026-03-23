import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

# Config
D_MODEL = 64
EPOCHS = 5

# Dataset
dataset = load_dataset("bentrevett/multi30k", split="train[:200]")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
VOCAB_SIZE = tokenizer.vocab_size


def tokenize(example):
    src = tokenizer(example["en"], padding="max_length", truncation=True, max_length=20)
    tgt = tokenizer(example["de"], padding="max_length", truncation=True, max_length=20)
    return {
        "src": torch.tensor(src["input_ids"]),
        "tgt": torch.tensor(tgt["input_ids"])
    }


dataset = dataset.map(tokenize)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.linear = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        return self.linear(x)


model = SimpleModel()

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(EPOCHS):
    total_loss = 0

    for sample in dataset:
        src = sample["src"].unsqueeze(0)
        tgt = sample["tgt"][0].unsqueeze(0)

        optimizer.zero_grad()

        output = model(src)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


print("Treinamento finalizado!")


sample = dataset[0]
src = sample["src"].unsqueeze(0)

output = model(src)
pred = torch.argmax(output, dim=1)

print("Teste de Overfitting:")
print("Entrada:", sample["src"][:10])
print("Saída prevista:", pred)
