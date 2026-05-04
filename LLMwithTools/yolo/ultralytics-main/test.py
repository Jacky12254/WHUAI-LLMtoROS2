from ultralytics import YOLO
 
if __name__=="__main__":
    
    pth_path=r"C:\Users\Agora\robohome\runs\train\exp2\weights\best.pt"
 
    test_path=r"C:\Users\Agora\robohome\data\dataset\images\test"
    # Load a model
    #model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model
 
    # Predict with the model
    results = model(test_path,save=True,conf=0.5)  # predict on an image