import os
import vtk
import qt
import slicer
import numpy as np
import torch
import torch.nn as nn
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import time

from OnlineDataGenerator import OnlineDataGenerator
from BiLSTMModel import BiLSTMModel as myModel


model_path=  "/Users/zhk/Desktop/fall2024/THESIS/models/best_bilstm_model.pth" #Replace with your model path

# ================================
# Slicer Module Definition
# ================================
class SON(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SON"
        self.parent.categories = ["Examples"]
        self.parent.helpText = "This module processes real-time transformation data."


class SONWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = SONLogic(widget=self)

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        #print the predicted class 
        self.predictedClassLabel = qt.QLabel("Latest Prediction: N/A")
        self.layout.addWidget(self.predictedClassLabel)

        # Start Receiver Button
        self.startReceiverButton = qt.QPushButton("Start Data Receiver")
        self.layout.addWidget(self.startReceiverButton)
        self.startReceiverButton.clicked.connect(self.onStartReceiverButton)

        # Stop Receiver Button
        self.stopReceiverButton = qt.QPushButton("Stop Data Receiver")
        self.layout.addWidget(self.stopReceiverButton)
        self.stopReceiverButton.clicked.connect(self.onStopReceiverButton)

    def onStartReceiverButton(self):
        print("Start Data Receiver button clicked.")
        self.logic.listAllNodes()
        with slicer.util.tryWithErrorDisplay("Failed to start data receiver.", waitCursor=True):
            self.logic.setupSequenceReceiver()

    def onStopReceiverButton(self):
        print("Stop Data Receiver button clicked.")
        self.logic.stopReceiver()
        self.logic.report_latency()



# ================================
#  Main Logic for Processing
# ================================
class SONLogic(ScriptedLoadableModuleLogic):
    def __init__(self, widget=None) -> None:
        self.widget = widget
        ScriptedLoadableModuleLogic.__init__(self)
        self.sequence_node = None
        self.observer_tag = None
        self.data_generator = OnlineDataGenerator(slice_width=60)
        self.model_path = model_path
        self.model = self.load_model()
        self.latency_log = []


    def listAllNodes(self):
        print("Listing all nodes in the scene:")
        node_collection = slicer.mrmlScene.GetNodes()
        node_collection.InitTraversal()
        node = node_collection.GetNextItemAsObject()
        while node:
            print(f"Node Name: {node.GetName()}, Type: {node.GetClassName()}")
            node = node_collection.GetNextItemAsObject()

    def setupSequenceReceiver(self):
        try:
            self.needle_node = slicer.util.getNode("NeedleToReference")
            self.probe_node = slicer.util.getNode("ProbeToReference")
            print(f"Nodes found: {self.needle_node.GetName()}, {self.probe_node.GetName()}")
        except slicer.util.MRMLNodeNotFoundException:
            print("Error: One or both transformation nodes not found.")
            self.listAllNodes()
            return

        if isinstance(self.needle_node, slicer.vtkMRMLLinearTransformNode) and \
           isinstance(self.probe_node, slicer.vtkMRMLLinearTransformNode):

            self.observer_tag = self.needle_node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onTransformReceived)
            print(f"Receiver set up for NeedleToReference and ProbeToReference.")

    def stopReceiver(self):
        if self.needle_node and self.probe_node and self.observer_tag:
            self.needle_node.RemoveObserver(self.observer_tag)
            print("Receiver stopped.")
        else:
            print("No active receiver to stop.")

    def onTransformReceived(self, caller, event):
        print(" New transform data received!")

        if not isinstance(self.needle_node, slicer.vtkMRMLLinearTransformNode) or \
           not isinstance(self.probe_node, slicer.vtkMRMLLinearTransformNode):
            print("Error: Needle or Probe transform node is missing or incorrect type.")
            return

        # Extract transformation matrices
        needle_matrix = vtk.vtkMatrix4x4()
        probe_matrix = vtk.vtkMatrix4x4()
        self.needle_node.GetMatrixTransformToParent(needle_matrix)
        self.probe_node.GetMatrixTransformToParent(probe_matrix)

        # Convert to NumPy arrays
        needle_np = slicer.util.arrayFromVTKMatrix(needle_matrix)
        probe_np = slicer.util.arrayFromVTKMatrix(probe_matrix)

        print(f"Received Needle matrix:\n{needle_np}")
        print(f"Received Probe matrix:\n{probe_np}")

        # Add transformed data to the generator
        self.data_generator.add_data(needle_np, probe_np)

        if len(self.data_generator) > 0:
            input_data = self.data_generator[-1].unsqueeze(0)
            processed_data = self.datagenOnline(input_data)

            prediction = self.classify_data(processed_data)
            print(f"✅ Prediction: {prediction}")
            if self.widget and hasattr(self.widget, 'predictedClassLabel'):
                self.widget.predictedClassLabel.setText(f"Latest Prediction: {prediction}")


    def datagenOnline(self, data):
        """
        Normalize online data.
        """
        all_slices = torch.tensor(self.data_generator.slices, dtype=torch.float32)
        global_mean = torch.mean(all_slices, dim=0)
        global_std = torch.std(all_slices, dim=0)
        global_std[global_std == 0] = 1e-8

        return (data - global_mean) / global_std

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            return None

        model = myModel()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        print(f"✅ Model loaded from: {self.model_path}")
        return model

    def classify_data(self, processed_data):
        if self.model is None:
            print("❌ No model loaded.")
            return "Unknown"

        with torch.no_grad():
            start_time = time.time()
            output = self.model(processed_data)
            latency = time.time() - start_time
            self.latency_log.append(latency)

        predicted_class = torch.argmax(output, dim=1).item()
        return "Expert " if predicted_class == 1 else "Novice "
    
    def report_latency(self):
        if not self.latency_log:
            print("No latency data collected.")
            return

        avg_latency = sum(self.latency_log) / len(self.latency_log)
        min_latency = min(self.latency_log)
        max_latency = max(self.latency_log)

        print(f"\n📊 Latency Summary over {len(self.latency_log)} predictions:")
        print(f"Average Latency: {avg_latency:.4f} seconds")
        print(f"Min Latency: {min_latency:.4f} seconds")
        print(f"Max Latency: {max_latency:.4f} seconds")
