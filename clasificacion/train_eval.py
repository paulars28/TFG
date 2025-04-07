from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt


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
    
    torch.save(model.state_dict(), model_name)
    print("\n✅ Modelo guardado como " + model_name)
    return train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist


def evaluate_model(test_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist):
    # Graficar loss y accuracy
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label='Training Loss', color='blue')
    plt.plot(test_loss_hist, label='Test Loss', color= 'orange')
    plt.title('Training/Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label='Training Accuracy', color='blue')
    plt.plot(test_acc_hist, label='Test Accuracy', color= 'orange')
    plt.title('Training/Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()