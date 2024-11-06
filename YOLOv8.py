from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt')

model.train(data='blood_train.yaml', epochs=10)
model.val()

