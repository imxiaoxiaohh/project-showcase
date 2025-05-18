
import cv2
import matplotlib.pyplot as plt
from preprocess import preprocess_image, visualize_symbols
from recognition import recognize_symbols
from postprocess import merge_dashes_to_equals, recognize_divide, dash_to_fraction, detect_subexp
from config import CLASSES 
import json

def run_inference(model, image_path, visualize=False):
    """
    Complete inference pipeline:
      1. Preprocess the image.
      2. Recognize symbols using the loaded model.
      3. Post-process the raw predictions.
    """
    # (Optional) display the original image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if visualize:
        plt.figure(figsize=(8, 4))
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.show()

    # Step 1: Preprocess the image to extract symbols
    symbols = preprocess_image(image_path)
    if visualize:
        visualize_symbols(symbols)

    # Step 2: Recognize symbols (prediction)
    predictions = recognize_symbols(symbols, model, top_k=3)

    predictions_with_divide = recognize_divide(predictions, CLASSES, size_threshold=30, vertical_threshold=60)

    # Step 3: Post-process the predictions
    updated_predictions = merge_dashes_to_equals(predictions_with_divide, CLASSES, x_threshold=30, w_threshold=50)
    
    final_results, fraction_info = dash_to_fraction(updated_predictions, CLASSES)
    subexp_info = detect_subexp(final_results)

    return final_results, fraction_info, subexp_info

def convert_output_to_json(final_results, fractions, subexp_info):
    """
    Convert the output list into json 
    """
    symbols_list = []

    for i, entry in enumerate(final_results):
        possible_syms, confidences, bbox_info = entry

        # construct a dictionary { symbol: confidence, ... }
        possible_symbols_conf = {
            sym: float(conf) for sym, conf in zip(possible_syms, confidences)
        }

        # bbox_info
        x, y, w, h, cx, cy, aspect_ratio = bbox_info
        
        #dictionary for bounding box data
        bounding_box = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "cx": cx,
            "cy": cy,
            "aspect_ratio": aspect_ratio
        }

        # symbol dictionary
        symbol_dict = {
            "index": i,
            "possible_symbols_confidence": possible_symbols_conf,
            "bounding_box": bounding_box
        }

        symbols_list.append(symbol_dict)

    # final dictionary
    output_dict = {
        "symbols": symbols_list,
        "fractions": fractions,
        "exponents": subexp_info
    }


    json_str = json.dumps(output_dict, indent=2)
    return json_str

if __name__ == '__main__':
    from recognition import load_model
    from decode import decode_base64_to_image_file

    model = load_model()  

    image_path = "D:/Bijlex/Math-recognition/validation/3.png"
    # base64_str = ""

    # image_path = decode_base64_to_image_file(base64_str)

    final_results, fractions, subexp_info = run_inference(model, image_path, visualize=True)

    print("processed prediction", final_results)
    print('frac', fractions)
    print('exp', subexp_info)
    json_output = convert_output_to_json(final_results, fractions, subexp_info)
    print(json_output)
    with open("output.json", "w") as f:
        f.write(json_output)

