# -*- coding: utf-8 -*-  
print("=== VERIFICACAO AMBIENTE ===")  
  
import sys  
print(f"Python: {sys.version}")  
print(f"Encoding: {sys.getdefaultencoding()}")  
  
try:  
    import mindspore as ms  
    print(f"MindSpore: {ms.__version__} ?")  
except:  
    print("MindSpore: ?")  
  
try:  
    import cv2  
    print(f"OpenCV: {cv2.__version__} ?")  
except:  
    print("OpenCV: ?")  
  
try:  
    import mediapipe as mp  
    print("MediaPipe: ?")  
except:  
    print("MediaPipe: ?")  
  
print("=== FIM VERIFICACAO ===")  
