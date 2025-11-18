import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ltn
import numpy as np

# ==========================================
# 1. Definição da CNN (Classificador Binário)
# ==========================================
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        # Arquitetura simples para processar imagens (ex: 32x32)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
            # Nota: Sem Sigmoide aqui, o ltn.Predicate cuida disso.
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 2. Dataset Simulado (Cães e Gatos)
# ==========================================
class DogCatDataset(Dataset):
    def __init__(self, num_samples=200):
        # Simula 200 imagens aleatórias para o código rodar sem precisar de arquivos externos
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 3, 32, 32) 
        # Metade Cachorro (1), Metade Gato (0)
        self.targets = torch.cat((torch.ones(num_samples//2), torch.zeros(num_samples - num_samples//2)))
        
        # Embaralha
        idx = torch.randperm(num_samples)
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Função para separar o batch (Necessário para a lógica do artigo)
def get_separated_batches(dataloader):
    for images, labels in dataloader:
        dog_imgs = images[labels == 1]
        cat_imgs = images[labels == 0]
        if len(dog_imgs) > 0 and len(cat_imgs) > 0:
            yield dog_imgs, cat_imgs

# ==========================================
# 3. Treinamento LTN (Implementação do Artigo)
# ==========================================
def train():
    # Configurações
    n_epochs = 20
    batch_size = 32
    lr = 0.001

    # Instancia modelo e dados
    dataset = DogCatDataset()
    standard_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CNN_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Componentes LTN
    Dog = ltn.Predicate(model)
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="∀")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    print("--- Iniciando Treinamento LTN ---")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        count = 0
        
        # Usa o gerador para pegar cães e gatos separados
        train_data_loader = get_separated_batches(standard_loader)
        
        for dog_imgs, cat_imgs in train_data_loader:
            optimizer.zero_grad()

            # Grounding das variáveis lógicas
            dog_var = ltn.Variable("dog", dog_imgs)
            cat_var = ltn.Variable("cats", cat_imgs)

            # Cálculo da Satisfação (Loss)
            # Regra 1: Todo dog é Dog. Regra 2: Todo cat NÃO é Dog.
            sat_agg = SatAgg(
                Forall(dog_var, Dog(dog_var)),
                Forall(cat_var, Not(Dog(cat_var)))
            )

            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
        
        if count > 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/count:.4f}")

if __name__ == "__main__":
    train()
