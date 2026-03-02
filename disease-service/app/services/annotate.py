from PIL import ImageDraw
import io

def draw_tipburn_boxes(pil_img, yolo_result):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return img

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, max(0, y1 - 15)), f"{c:.2f}", fill="red")

    return img

def pil_to_png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
