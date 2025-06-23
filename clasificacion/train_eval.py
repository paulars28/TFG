from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, matthews_corrcoef, roc_curve, precision_recall_curve, auc

import os
import seaborn as sns


def train_eval_loop(train_loader, test_loader, model, epochs, optimizer, criterion, device, model_name):
    # Historial de métricas
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    for epoch in range(epochs):
        # ----- Entrenamiento -----
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = running_loss / len(train_loader)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # ----- Evaluación -----
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                preds = (outputs > 0.5).float()
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_loss = test_loss / len(test_loader)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    torch.save(model.state_dict(), model_name + ".pth") 
    print("\n Modelo guardado como " + model_name)
    return train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist


def evaluate_model(test_loader, model, device, model_name):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            probs = torch.sigmoid(outputs).squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            
    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)


    print("\n Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    result_file = model_name + "_metricas.txt"
    with open(result_file, "w") as f:
        f.write(f"Evaluation Metrics for model: {model_name}\n\n")
        f.write(f"Accuracy:           {accuracy:.4f}\n")
        f.write(f"Precision:          {precision:.4f}\n")
        f.write(f"Recall:             {recall:.4f}\n")
        f.write(f"F1 Score:           {f1:.4f}\n")
        f.write(f"ROC AUC:            {roc_auc:.4f}\n")
        f.write(f"PR AUC:             {pr_auc:.4f}\n")
        f.write(f"Matthews CorrCoef:  {mcc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")

    save_eval_arrays(model_name, all_labels, all_probs)
    plot_eval_curves(all_labels, all_probs, model_name)
    plot_confusion_matrix_heatmap(all_labels, all_preds, model_name)
    


def plot_eval_curves(all_labels, all_probs, model_name):

    # Curva ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1)
    plt.xlabel("Tasa de Falsos Positivos", fontsize=12)
    plt.ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    plt.title("Curva ROC", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(model_name + "_curva_ROC.png", dpi=300)
    plt.close()

    # Curva Precisión-Recall
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    plt.figure()
    plt.plot(rec, prec, label=f"Curva Precisión-Recall (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall (Sensibilidad)", fontsize=12)
    plt.ylabel("Precisión", fontsize=12)
    plt.title("Curva Precisión-Recall", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(model_name + "_curva_PR.png", dpi=300)
    plt.close()



COLORES_MODELO = {
    "KAN": "#E07B39",  # Naranja pastel oscuro
    "MLP": "#6BAED6"   # Azul pastel oscuro
}

def plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, model_name):

    color = COLORES_MODELO.get(model_name, "gray")

    plt.figure(figsize=(10, 4))
    plt.title(f"Curvas de pérdida y precisión – {model_name}", fontsize=16)

    # Subgráfico: Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label='Pérdida en entrenamiento', color=color, linewidth=0.75, linestyle='-')
    plt.plot(test_loss_hist, label='Pérdida en validación', color=color, linewidth=0.75, linestyle='--')
    plt.title('Evolución de la pérdida', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    # Subgráfico: Precisión
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label='Precisión en entrenamiento', color=color, linewidth=0.75, linestyle='-')
    plt.plot(test_acc_hist, label='Precisión en validación', color=color, linewidth=0.75, linestyle='--')
    plt.title('Evolución de la precisión', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{model_name}_pérdida_precisión.png", dpi=300)
    plt.show()


def plot_confusion_matrix_heatmap(all_labels, all_preds, model_name, save=True):

    cm = confusion_matrix(all_labels, all_preds)


    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 12})
    
    plt.xlabel("Etiqueta predicha", fontsize=12)
    plt.ylabel("Etiqueta real", fontsize=12)
    plt.title(f"Matriz de confusión – {model_name}", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(f"{model_name}_matriz_confusion.png", dpi=300)
    plt.close()


#funcion para guardar arrays durante la evaluacion y despues usarlos en compare.py
def save_eval_arrays(model_name, all_labels, all_probs, output_dir="./eval_arrays"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, model_name + "_eval.npz")
    np.savez(path, labels=all_labels, probs=all_probs)