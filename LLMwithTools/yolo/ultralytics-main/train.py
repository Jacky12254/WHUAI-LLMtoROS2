import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'C:\Users\Agora\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
