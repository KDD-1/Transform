from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt')

model.predict('954/test/images/BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg', save=True)
