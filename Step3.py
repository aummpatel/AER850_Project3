# Step 3: YOLOv11 Evaluation
from ultralytics import YOLO


# 1. Load the TRAINED model weights
# The path is usually: project_path / name / 'weights' / 'best.pt'
# Based on your settings above, it should be here:
trained_model_path = "Project 3 Outputs/Model Outputs/Model150_8_1280/weights/best.pt"

# Load the new, smart model
best_model = YOLO(trained_model_path) 

# Paths to evaluation images
ardmega_path = "Project 3 Data/Project 3 Data/data/data/evaluation/ardmega.jpg"
arduno_path = "Project 3 Data/Project 3 Data/data/data/evaluation/arduno.jpg"
rasppi_path = "Project 3 Data/Project 3 Data/data/data/evaluation/rasppi.jpg"

# 2. Run predictions using 'best_model', NOT 'model'
print("Evaluating Arduino Mega...")
results_arduinomega = best_model.predict(
    source = ardmega_path,        
    conf = 0.005,                
    save = True,                
    line_width = 4
)

print("Evaluating Arduino Uno...")
results_arduinoUno = best_model.predict(
    source=arduno_path,
    conf=0.005,               
    save=True,              
    line_width = 1
    )

print("Evaluating Raspberry Pi...")
results_rasberrypi = best_model.predict(
    source=rasppi_path,          
    conf=0.005,               
    save=True,               
    line_width = 3
    )