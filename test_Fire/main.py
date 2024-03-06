from ultralytics import YOLO

model = YOLO("best.pt")

results = model(source='test3.mp4', show=True, conf=0.4, save=True)
