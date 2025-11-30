#%% Step 2: YOLOv11 Training

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
# Defining Dataset Path as well as Output destination path
ds_path = "Project 3 Data/Project 3 Data/data/data/data.yaml"
output_path = "Project 3 Outputs/Model Outputs" 
model = YOLO("yolo11n.pt")

# Smaller Label Annotations
Annotator.text_color = (255, 255, 255)
Annotator.box_line_width = 1
Annotator.font_size = 0.1

if __name__ == "__main__":
    results = model.train(
    data=ds_path,
    epochs = 150,
    batch = 8,
    imgsz = 1200,
    workers = 8,
    amp = True,
    device = 0,
    name = "Model150_8_1280",
    project = output_path
    )

# %%

metrics = model.val(data=ds_path, split="test")
metrics.box.map  
metrics.box.map50  
metrics.box.map75  
metrics.box.maps



# %%
# Step 3: YOLOv11 Evaluation
# Paths to evaluation images
ardmega_path = "Project 3 Data/Project 3 Data/data/data/evaluation/ardmega.jpg"
arduno_path = "Project 3 Data/Project 3 Data/data/data/evaluation/arduno.jpg"
rasppi_path = "Project 3 Data/Project 3 Data/data/data/evaluation/rasppi.jpg"

# Prediction for Ardmega
results_arduinomega = model.predict(
    source = ardmega_path,  # Path to Ardmega image
    imgsz = 2048,               # Image size
    conf = 0.35,                # Confidence threshold
    save = True,                # Save the prediction
    line_width = 3
)

# Prediction for Ardu
results_arduinoUno = model.predict(
    source=arduno_path,  # Path to Ardu image
    imgsz = 2048,            # Image size
    conf=0.35,               # Confidence threshold
    save=True,              # Save the prediction
    line_width = 3
    )

# Prediction for Raspi
results_rasberrypi = model.predict(
    source=rasppi_path,  # Path to Raspi image
    imgsz = 2048,            # Image size
    conf=0.35,               # Confidence threshold
    save=True,               # Save the prediction
    line_width = 3
    )

# systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"