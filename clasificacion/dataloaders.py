from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
		


class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
		


def get_dataset_train_val(batch_size):
	df = pd.read_csv("Heart Prediction Quantum Dataset.csv")
	X = df.drop("HeartDisease", axis=1).values
	y = df["HeartDisease"].values

	# Normalización
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Datasets y Dataloaders
	train_dataset = HeartDiseaseDataset(X_train, y_train)
	test_dataset = HeartDiseaseDataset(X_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)
	return train_loader, test_loader