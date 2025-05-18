import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import MathSymbolCNN
from preprocess import preprocess_image

def load_model(model_path="models/math_cnn.pth"):
    num_classes = 44  # Adjust as necessary
    model = MathSymbolCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def recognize_symbols(symbols, model, top_k=3):
    predictions = []

    for symbol_data in symbols:
        if isinstance(symbol_data, tuple) and len(symbol_data) == 2:
            symbol, position = symbol_data  # Unpack image and position
        else:
            print(f"Skipping invalid symbol format: {symbol_data}")
            continue

        # Convert to PyTorch Tensor and Ensure Correct Shape
        symbol = torch.tensor(symbol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,28,28)

        # Ensure Background Matches Dataset
        if torch.mean(symbol) < 0.5:
            symbol = 1 - symbol

        with torch.no_grad():
            output = model(symbol)
            probabilities = F.softmax(output, dim=1)
            top_probs, top_labels = torch.topk(probabilities, top_k, dim=1)

            top_labels = top_labels.squeeze(0).tolist()
            top_probs = top_probs.squeeze(0).tolist()

            predictions.append((top_labels, top_probs, position))

    return predictions

def predict(image_path, model):
    symbols = preprocess_image(image_path)
    return recognize_symbols(symbols, model)

