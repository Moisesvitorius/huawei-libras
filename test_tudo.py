import cv2  
import mediapipe as mp  
import numpy as np  
import mindspore as ms  
print("? OpenCV:", cv2.__version__)  
print("? MediaPipe: OK")  
print("? MindSpore:", ms.__version__)  
print("? NumPy:", np.__version__)  
x = ms.Tensor([1, 2, 3, 4, 5])  
print("? Tensor test:", x.asnumpy())  
