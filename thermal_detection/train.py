from ultralytics import YOLO

dataset_train = "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/config/data_thermal.yaml"

model = YOLO("yolov8n.yaml")

results = model.train(data=dataset_train, epochs=5, classes=[1, 3])
