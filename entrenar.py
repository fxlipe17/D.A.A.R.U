from ultralytics import YOLO


model = YOLO('yolov8n.pt')


model.train(
    data='datos/data.yaml', 
    epochs=50,
    imgsz=640,
    batch=8,                
    name='mi_detector',
    device='cpu'             
)

print("¡Entrenamiento terminado!")