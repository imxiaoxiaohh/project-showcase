import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_with_padding(symbol, target_size=(28, 28)):
    """
    Resizes an image while maintaining aspect ratio and adding padding to fit target size.
    Keeps background black (0) and symbol white (255).
    """
    h, w = symbol.shape[:2]
    aspect_ratio = w / h

    # Determine new size while keeping aspect ratio
    if w > h:  
        new_w = target_size[0] - 4  
        new_h = int(new_w / aspect_ratio)
    else: 
        new_h = target_size[1] - 4  
        new_w = int(new_h * aspect_ratio)

    # Resize with preserved aspect ratio
    resized = cv2.resize(symbol, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas and center the resized image
    canvas = np.zeros(target_size, dtype=np.uint8)  # Black background
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def preprocess_image(image_path):
    """
    Preprocesses an image to extract symbols while preserving their aspect ratio.
    Adds detailed positional information: (x, y, w, h, cx, cy).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img_blur = cv2.GaussianBlur(img, (5,5), 0)  
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  

    # Morphological operation to improve contours
    kernel = np.ones((3,3), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):  # Sort left to right
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box

        # Expand bounding box slightly
        pad = 5
        x, y, w, h = max(0, x-pad), max(0, y-pad), w+2*pad, h+2*pad

        symbol = img_bin[y:y+h, x:x+w]  

        # Resize with padding
        symbol = resize_with_padding(symbol, (28, 28))

        if np.mean(symbol) < 128:  
            symbol = cv2.bitwise_not(symbol)  

        # Normalize 
        symbol = symbol.astype(np.float32) / 255.0

        # Compute additional position info
        cx = x + w // 2  
        cy = y + h // 2 
        aspect_ratio = w / h  # Width-to-height ratio

        symbols.append((symbol, (x, y, w, h, cx, cy, aspect_ratio)))

    return symbols

def visualize_symbols(symbols):
    fig, axes = plt.subplots(1, len(symbols), figsize=(12, 4))

    for i, (symbol, _) in enumerate(symbols):
        axes[i].imshow(symbol, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Symbol {i}")

    plt.show()