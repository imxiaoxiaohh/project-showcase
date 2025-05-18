import base64, tempfile
import cv2
import numpy as np

def decode_base64_to_image_file(base64_str):
    """
    Decodes a base64 image string (with optional data URL prefix),
    composites any transparency onto white, and writes it to a
    temporary file in its native format.
    Returns the path to the temp file.
    """
    # strip off data URL if it’s there
    if base64_str.startswith("data:image/"):
        base64_str = base64_str.split(",", 1)[1]

    # fix padding
    pad = len(base64_str) % 4
    if pad:
        base64_str += "=" * (4 - pad)

    # Base64 - bytes - numpy buffer
    img_bytes = base64.b64decode(base64_str)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)

    # decode with OpenCV 
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image data")

    # if there’s an alpha channel, composite it onto white
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = (a.astype(float) / 255.0)[..., None]
        white = np.ones_like(img[..., :3], dtype=np.uint8) * 255
        rgb = img[..., :3].astype(float)
        img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    # write out as a PNG so imread will always work
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, img)
    return tmp.name