import gradio as gr
import cv2
import mediapipe as mp
from gradio_webrtc import WebRTC

# Configuración de detección de rostros con MediaPipe
mp_face_detection = mp.solutions.face_detection

def process_img(img, face_detection, blur_value):
    """Anonymiza las caras en la imagen mediante desenfoque."""
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Calcular las coordenadas del bounding box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Desenfocar las caras detectadas con el valor del slider
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (blur_value, blur_value))

    return img

# Función principal para la detección y desenfoque de rostros en Gradio
def detection(image, conf_threshold=0.5, blur_value=30):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=conf_threshold) as face_detection:
        processed_image = process_img(image, face_detection, blur_value)
    return cv2.resize(processed_image, (500, 500))

css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Face Anonymization Webcam Stream (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Anonymize Faces in Real-Time
        </h3>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=None)
            conf_threshold = gr.Slider(
                label="Detection Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.50,
            )
            blur_value = gr.Slider(
                label="Blur Intensity",
                minimum=5,
                maximum=100,
                step=5,
                value=30,
            )

        image.stream(
            fn=detection, inputs=[image, conf_threshold, blur_value], outputs=[image], time_limit=10
        )

if __name__ == "__main__":
    demo.launch(share=True)

