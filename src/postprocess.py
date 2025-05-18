import numpy as np

def merge_dashes_to_equals(predictions, classes, x_threshold=30, w_threshold=50):
    """
    Recognize dash signs written as two vertically aligned equal (=).
    """

    updated_predictions = []

    num_preds = len(predictions)
    i = 0

    while i < num_preds - 1:
        top_classes_1, confidences_1, location_info_1 = predictions[i]
        top_classes_2, confidences_2, location_info_2 = predictions[i + 1]

        # Check if both adjacent symbols contain '-' in their top predictions
        contains_dash_1 = any(classes[cls] == '-' for cls in top_classes_1)
        contains_dash_2 = any(classes[cls] == '-' for cls in top_classes_2)

        if contains_dash_1 and contains_dash_2:
            loc1 = location_info_1
            loc2 = location_info_2

            if isinstance(loc1, list):
                loc1 = loc1[0]
            if isinstance(loc2, list):
                loc2 = loc2[0]

            x_diff = abs(loc2[0] - loc1[0])
            w_diff = abs(loc2[2] - loc1[2])

            print(f"Checking adjacent symbols {i} and {i+1} for equality merge")
            print("x_diff:", x_diff, "w_diff:", w_diff)

            if x_diff < x_threshold and w_diff < w_threshold:
                print("Merging adjacent dashes into '=' symbol")

                equality_index = classes.index('=')

                # Compute new location info for '='
                x_min = min(loc1[0], loc2[0])
                y_min = min(loc1[1], loc2[1])
                x_max = max(loc1[0] + loc1[2], loc2[0] + loc2[2])
                y_max = max(loc1[1] + loc1[3], loc2[1] + loc2[3])

                new_location = (
                    x_min, y_min, x_max - x_min, y_max - y_min,
                    (x_min + x_max) // 2, (y_min + y_max) // 2,
                    (x_max - x_min) / (y_max - y_min)
                )

                new_confidence = (confidences_1[0] + confidences_2[0]) / 2

                updated_predictions.append((
                    [equality_index],  # Only one '=' symbol
                    [new_confidence],  # Updated confidence
                    (new_location)  # Updated location
                ))

                i += 2  # Skip next symbol since we merged
                continue

        updated_predictions.append(predictions[i])
        i += 1

    if i == num_preds - 1:
        updated_predictions.append(predictions[i])

    return updated_predictions

def recognize_divide(predictions, classes, size_threshold=30, vertical_threshold=60):
    """
    Recognize division signs written as two vertically aligned dots (:).
    Returns:
        updated_predictions: List of tuples with division signs recognized
    """
    if len(predictions) < 2:
        return predictions  # Not enough symbols to find division signs
    
    # Try to find the ":" (divide) symbol in the classes list
    divide_index = -1
    if ":" in classes:
        divide_index = classes.index(":")
    elif "divide" in classes:
        divide_index = classes.index("divide")

    if divide_index == -1:
        print("Division symbol ':' not found in classes, adding to classes list")
        classes.append(":")
        divide_index = len(classes) - 1
    
    updated_predictions = []
    i = 0
    
    while i < len(predictions) - 1:
        _, _, loc1 = predictions[i]
        _, _, loc2 = predictions[i + 1]
        
        # Ensure we have the correct location format
        if isinstance(loc1, list):
            loc1 = loc1[0]
        if isinstance(loc2, list):
            loc2 = loc2[0]
        
        # Extract width, height and positions
        x1, y1, w1, h1, cx1, cy1, _ = loc1
        x2, y2, w2, h2, cx2, cy2, _ = loc2
        
        # Check if both symbols are small enough to be dots
        is_small1 = w1 < size_threshold and h1 < size_threshold
        is_small2 = w2 < size_threshold and h2 < size_threshold
        
        # Check if they are vertically aligned
        is_aligned = abs(cx1 - cx2) < size_threshold
        
        print(f"Checking symbols {i} and {i+1} for division sign:")
        print(f"  Sizes: ({w1}x{h1}), ({w2}x{h2})")
        print(f"  Horizontal alignment: {abs(cx1 - cx2)}")
        
        if is_small1 and is_small2 and is_aligned:
            # Merge the two dots into a division sign
            print(f"  Recognized division sign ':'")
            
            # Calculate new bounding box for combined symbol
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            
            new_w = x_max - x_min
            new_h = y_max - y_min
            new_cx = (x_min + x_max) // 2
            new_cy = (y_min + y_max) // 2
            aspect_ratio = new_w / new_h if new_h > 0 else 1.0
            
            new_loc = (x_min, y_min, new_w, new_h, new_cx, new_cy, aspect_ratio)
            
            # Average the confidences for the combined symbol
            avg_conf = 0.95  # High confidence for the merge
            
            # Create new prediction for the division sign
            updated_predictions.append((
                [divide_index],  # Division symbol index
                [avg_conf],      # High confidence
                new_loc          # Updated location
            ))
            
            i += 2  # Skip the next symbol
            continue
        
        updated_predictions.append(predictions[i])
        i += 1
    
    # Add the last symbol if we haven't processed it
    if i == len(predictions) - 1:
        updated_predictions.append(predictions[i])
    
    return updated_predictions

def dash_to_fraction(predictions, referred_classes):
    dash_index = referred_classes.index('-')
    processed_predictions = []
    dash_candidates = []
    fractions = []  # Store detected fractions with their numerator & denominator

    for pred in predictions:
        top_classes, confidences, (x, y, w, h, cx, cy, aspect_ratio) = pred
        processed_predictions.append([
            top_classes,
            confidences,
            (x, y, w, h, cx, cy, aspect_ratio)
        ])

        # Check if the dash is in the top classes
        if dash_index in top_classes:
            dash_candidates.append({
                'index': len(processed_predictions) - 1,
                'cx': cx,
                'cy': cy,
                'w': w,
                'conf': confidences[top_classes.index(dash_index)],  # Confidence of dash
            })

    # Process each detected dash
    for dash in dash_candidates:
        dash_index = dash['index']
        dash_cx = dash['cx']
        dash_cy = dash['cy']
        dash_w = dash['w']
        dash_conf = dash['conf']

        # **Increase horizontal tolerance**
        min_x = dash_cx - 0.5 * dash_w  # 70% width centered around the dash
        max_x = dash_cx + 0.5 * dash_w

        numerator = []
        denominator = []

        for i, symbol in enumerate(processed_predictions):
            if i == dash_index:
                continue
            _, _, (sx, sy, sw, sh, scx, scy, sasp) = symbol

            if min_x <= scx <= max_x:
                # **Add a vertical gap threshold (e.g., 10 pixels)**
                if scy < dash_cy:
                    numerator.append(i)
                elif scy > dash_cy:
                    denominator.append(i)

        if numerator and denominator:
            processed_predictions[dash_index][0][0] = -1  # Indicating \frac with -1
            processed_predictions[dash_index][1][0] = dash_conf  # Assigning confidence
            fractions.append({
                'frac_index': dash_index,
                'numerator': numerator,
                'denominator': denominator
            })


    # Replace -1 with "\frac" in predictions
    for result in processed_predictions:
        result[0] = ["\\frac" if idx == -1 else referred_classes[idx] for idx in result[0]]

    return processed_predictions, fractions


def detect_subexp(processed_predictions):
    """
    Detect subscript and exponent relationships 
    Returns:
       subexp_info: list of dicts:
            [
              {"base_idx": i, "sub_idx": s, "exp_idx": e},
              ...
            ]
    """


    subexp_temp = [{"subscript": None, "exponent": None} for _ in processed_predictions]

    #thresholds 
    HORIZONTAL_EXPANSION   = 0.5   # how far right we allow
    BASELINE_FACTOR        = 0.7   # height used to compute baseline
    VERTICAL_OFFSET_RATIO  = 0.5   # height for sub/exp detection

    for i, base_sym in enumerate(processed_predictions):
        base_classes, base_confs, (bx, by, bw, bh, bcx, bcy, basp) = base_sym

        # baseline
        base_line = by + BASELINE_FACTOR * bh

        # strict to the right side of the base symbol
        left_limit  = bx + bw  # right edge of the base
        right_limit = bx + bw * (1.0 + HORIZONTAL_EXPANSION)

        # Vertical threshold for sub/exp
        vertical_margin = bh * VERTICAL_OFFSET_RATIO

        for j, test_sym in enumerate(processed_predictions):
            if i == j:
                continue

            t_classes, t_confs, (tx, ty, tw, th, tcx, tcy, tasp) = test_sym

            # Check if within horizontal range
            if not (left_limit <= tcx <= right_limit):
                continue

            vertical_diff = tcy - base_line

            # Decide exponent vs subscript
            if vertical_diff < -vertical_margin:
                # above baseline -> exponent
                if subexp_temp[i]["exponent"] is None:
                    subexp_temp[i]["exponent"] = j
            elif vertical_diff > vertical_margin:
                # below baseline -> subscript
                if subexp_temp[i]["subscript"] is None:
                    subexp_temp[i]["subscript"] = j

    subexp_info = []
    for i, subexp_dict in enumerate(subexp_temp):
        has_sub = subexp_dict["subscript"] is not None
        has_exp = subexp_dict["exponent"]  is not None
        if has_sub or has_exp:
            entry = {"base_idx": i}
            if has_sub:
                entry["sub_idx"] = subexp_dict["subscript"]
            if has_exp:
                entry["exp_idx"] = subexp_dict["exponent"]
            subexp_info.append(entry)

    return subexp_info