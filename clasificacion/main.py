import torch
import torch.optim as optim
import yaml

from dataloaders import *
from models import *
from train_eval import *

def load_config(file_path="config.yaml"):
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

cfg = load_config(file_path="config.yaml")

batch_size = cfg["batch_size"]
train_loader, test_loader = get_dataset_train_val(batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg["select_model"] == "mlp":
    model = DiabetesMLP().to(device)
elif cfg["select_model"] == "kan":
    model = DiabetesKAN().to(device)

optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
criterion = nn.BCELoss()
epochs = cfg["epochs"]
model_name = cfg["model_name"]

train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = train_eval_loop(train_loader, test_loader, model, epochs, optimizer, criterion, device, cfg["model_name"])
evaluate_model(test_loader, model, device, model_name)
plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist)




