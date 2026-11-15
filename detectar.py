import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import pyttsx3

# Configuración
MODEL_PATH  = 'runs/detect/mi_detector/weights/best.pt'
CONFIDENCE  = 0.5
SAVE_DIR    = 'evidencia/'
COOLDOWN    = 5  # segundos entre alertas

os.makedirs(SAVE_DIR, exist_ok=True)

# Cargar modelo y voz
model  = YOLO(MODEL_PATH)
voz    = pyttsx3.init()
voz.setProperty('rate', 150)

last_alert = 0

def alerta(frame, detecciones):
    global last_alert
    now = time.time()
    if now - last_alert < COOLDOWN:
        return
    last_alert = now
    # Borde rojo
    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
    # Texto en pantalla
    cv2.putText(frame, f"BASURA DETECTADA: {len(detecciones)} objeto(s)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    # Guardar evidencia
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{SAVE_DIR}basura_{ts}.jpg", frame)
    # Alerta de voz
    voz.say("Atención, se detectó basura")
    voz.runAndWait()

# Iniciar cámara
cap = cv2.VideoCapture(0)
print("Cámara iniciada. Presiona Q para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results     = model(frame, conf=CONFIDENCE, verbose=False)
    detecciones = results[0].boxes
    annotated   = results[0].plot()

    if len(detecciones) > 0:
        alerta(annotated, detecciones)

    cv2.imshow("Detector de Basura", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()