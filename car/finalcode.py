import torch
import torchvision
from torch2trt import TRTModule
from torch2trt import torch2trt
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import time

device = torch.device('cuda')

# Load the optimized model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('path/to/your/model'))

print("start")
# Preprocessing function
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

normalize = torchvision.transforms.Normalize(mean, std)

#serial communication setup 
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)

#define uart send command
def send_command(command):
    try:
        ser.write(("#P=" + str(command) + '\n').encode())  # Convert the command to a string before encoding
        print(f"Sent command: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=224,
    capture_height=224,
    display_width=224,
    display_height=224,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Initialize the camera using OpenCV
#window
camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)  # Assuming the camera is connected to /dev/video0
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

# Initialize the robot
#from jetbot import Robot
#robot = Robot()

# Speed setting
base_speed = 0.3

avg_count=0
prob_buff = [10,0,0,0,0]

def update(frame):
    global robot
    global prob_buff 
    global avg_count
    x = preprocess(frame)
    y = model_trt(x)
    
    # Apply softmax to normalize the output
    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])
    prob_percent = prob_blocked*100
    flt_percent = int(prob_percent)
#    str_percent= str(int_percent).zfill(3)
    prob_buff[avg_count]=flt_percent
    avg_count = avg_count+1
#    send_command(str_percent)    
    if avg_count>=3:
        flt_percent = int(prob_buff[0] + prob_buff[1]+ prob_buff[2])/3
#        print("int_percent=",int_percent)
        int_percent= int(flt_percent)
        str_percent = str(int_percent).zfill(3)
#        print("str_percent=",str_percent)
        send_command(str_percent)
        avg_count=0
    #print(prob_percent)
    

# Main loop
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Update robot control
        update(frame)
        
        # Display the frame
#        cv2.imshow('Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    #robot.stop()

