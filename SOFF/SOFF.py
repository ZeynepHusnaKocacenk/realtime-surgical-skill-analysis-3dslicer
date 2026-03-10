import os
from sklearn.model_selection import KFold
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import qt
import vtk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from BiLSTMModel import BiLSTMModel as myModel
from OfflineDataGenerator import OfflineDataGenerator, read_mha_file

# =========================
# Slicer Module: SOFF
# =========================
class SOFF(ScriptedLoadableModule):
    def __init__(self, parent):
        parent.title = "SOFF"
        parent.categories = ["Examples"]
        parent.helpText = "Offline trainer and real-time BiLSTM inference."
        self.parent = parent


class SOFFWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = SOFFLogic()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.trainButton = qt.QPushButton("Train BiLSTM Model")
        self.layout.addWidget(self.trainButton)
        self.trainButton.connect('clicked(bool)', self.onTrainModel)

        self.retrainCheckbox = qt.QCheckBox("Force Retrain (Overwrite Existing Model)")
        self.layout.addWidget(self.retrainCheckbox)

        self.runButton = qt.QPushButton("Run Model on .seq.mha File")
        self.layout.addWidget(self.runButton)
        self.runButton.connect('clicked(bool)', self.onRunModel)
        self.runButton.setEnabled(os.path.exists(self.logic.save_path))

        self.statusLabel = qt.QLabel("Status: Idle")
        self.layout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

    def onTrainModel(self):
        self.statusLabel.setText("Training started...")
        slicer.app.processEvents()
        force_retrain = self.retrainCheckbox.isChecked()
        self.logic.run_training(force_retrain=force_retrain)
        self.statusLabel.setText("Training complete. Model saved!")
        self.runButton.setEnabled(True)

    def onRunModel(self):
        fileDialog = qt.QFileDialog()
        fileDialog.setNameFilter("MHA files (*.mha)")
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
        if not fileDialog.exec_():
            return
        selected_files = fileDialog.selectedFiles()
        if not selected_files:
            return

        mha_path = selected_files[0]
        self.statusLabel.setText("Running model on: " + os.path.basename(mha_path))
        slicer.app.processEvents()

        self.logic.run_inference(mha_path)
        self.statusLabel.setText("Model inference complete.")


class SOFFLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.train_dir = "/Users/zhk/Desktop/fall2024/THESIS/data" #replace with your dataset path
        model_dir = "/Users/zhk/Desktop/fall2024/THESIS/models"
        os.makedirs(model_dir, exist_ok=True)
        self.save_path = os.path.join(model_dir, "best_bilstm_model.pth")

    def run_training(self, force_retrain=False, k_folds=5):
        if os.path.exists(self.save_path) and not force_retrain:
            print("✅ Model already exists. Skipping training.")
            return

        full_dataset = OfflineDataGenerator(self.train_dir, augment=True)
        kfold = KFold(n_splits=k_folds, shuffle=True)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
            print(f"\n📁 Fold {fold + 1}/{k_folds}")

            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            val_subset = torch.utils.data.Subset(full_dataset, val_idx)

            model = myModel()
            trained_model = self.train_model(
                model=model,
                train_dataset=train_subset,
                val_dataset=val_subset,
                save_path=self.save_path,  # Can be changed per fold if saving multiple
                batch_size=32,
                num_epochs=10,
                learning_rate=0.0003,
                weight_decay=1e-5,
                early_stopping_patience=3
            )

            val_loss, val_accuracy = self.evaluate_model(trained_model, val_subset)
            fold_results.append((val_loss, val_accuracy))
            print(f"📊 Fold {fold + 1} Results — Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")

        avg_acc = sum(acc for _, acc in fold_results) / k_folds
        print(f"\n🏁 Average K-Fold Accuracy: {avg_acc:.2f}%")

    def run_inference(self, mha_path):
        model = myModel()
        model.load_state_dict(torch.load(self.save_path))
        model.eval()

        data = read_mha_file(mha_path)
        if data is None:
            print("❌ Failed to parse .mha file.")
            return

        slices = [data[i:i + 30] for i in range(len(data) - 30 + 1)]
        if not slices:
            print("⚠️ Not enough frames in .mha file for slicing.")
            return

        sequence = torch.tensor(slices, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(sequence)
            predicted = torch.argmax(outputs, dim=1)
            final_prediction = int(torch.mode(predicted).values.item())
            label_map = {0: "Novice", 1: "Expert"}
            print("📄 Predicted Class:", label_map[final_prediction])

    def train_model(self, model, train_dataset, val_dataset, save_path, batch_size, num_epochs, learning_rate, weight_decay, early_stopping_patience):
        print("📦 Loading data and initializing training loop...")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss, correct_train, total_train = 0, 0, 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train

            model.eval()
            val_loss, correct_val, total_val = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"⚠️ Early stopping triggered at epoch {epoch + 1}.")
                    break

        print("✅ Training complete!")
        return model

    def evaluate_model(self, model, test_dataset, batch_size=32):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        correct, total, test_loss = 0, 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total if total > 0 else 0
        return test_loss, accuracy
