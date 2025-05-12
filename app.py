import gradio as gr
from ultralytics import YOLO
import numpy as np

# Load the classification model
model = YOLO("best.pt")

def process_image(image):
    # Get predictions
    results = model(image)
    result = results[0]

    print("Processing image...")
    print(f"Result: {result}")

    detections = []
    if result.probs is not None:
        probs = result.probs.data.cpu().numpy()  # <--- اصلاح این خط
        class_names = result.names
        for i, prob in enumerate(probs):
            if prob > 0.01:
                detections.append((class_names[i], float(prob)))

    detections.sort(key=lambda x: x[1], reverse=True)

    if detections:
        top_class, top_conf = detections[0]
        output_text = f"🔝 Highest Confidence Detection: {top_class} ({top_conf:.2f})\n\n"
        output_text += "📋 All Detections:\n"
        for class_name, confidence in detections:
            output_text += f"{class_name}: {confidence:.2f}\n"
    else:
        output_text = "❌ No detections found"

    return image, output_text

# Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Input Image"),
        gr.Textbox(label="Detections", lines=8)
    ],
    title="Car Classification",
    description="Upload a car image to classify it. Shows the class with highest confidence on top."
)

if __name__ == "__main__":
    print("Starting application...")
    demo.launch(server_name="127.0.0.1", server_port=8080)
