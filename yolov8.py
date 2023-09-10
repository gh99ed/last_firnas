import shutil
from ultralytics import YOLO
import os

class CustomYOLOv8:
    def __init__(self, model_weights_path):
        # Load your custom YOLOv8 model weights
        self.model = YOLO(model_weights_path)

    def predict_image(self, image,filename):
        results =self.model.predict(source=image, save = True, project="STATIC", name="PREDICTED")
        # Show the results
        result = results[0]
        output = []
        for box in result.boxes:
            x1, y1, x2, y2 = [
              round(x) for x in box.xyxy[0].tolist()
            ]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            output.append([
              x1, y1, x2, y2, result.names[class_id], prob
            ])
        predicted_classes = []
        for i in output:
          predicted_classes.append(i[4])
          predicted_classes_num = len(predicted_classes)
        source = 'STATIC/PREDICTED'
        destination = 'STATIC/TEMP'
        src_path = os.path.join(source, filename)
        dst_path = os.path.join(destination, filename)
        shutil.move(src_path, dst_path)
        shutil.rmtree('STATIC/PREDICTED/')
        return predicted_classes, predicted_classes_num