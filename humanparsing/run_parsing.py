from pathlib import Path
import os
import onnxruntime as ort
from parsing_api import onnx_inference
import torch
from PIL import Image
import numpy as np


class Parsing:
    def __init__(self, atr_model_path, lip_model_path):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(atr_model_path,
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(lip_model_path,
                                                sess_options=session_options, providers=['CPUExecutionProvider'])


    def __call__(self, input_image):
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
    

if __name__ == '__main__':
    print("Parsing module is running.")
    atr_model_path = "/home/trgtuan/OneDrive/My Git/ComfyUI-OOTDiffusion-MaskOnly/checkpoints/parsing_atr.onnx"
    lip_model_path = "/home/trgtuan/OneDrive/My Git/ComfyUI-OOTDiffusion-MaskOnly/checkpoints/parsing_lip.onnx"
    parsing = Parsing(atr_model_path, lip_model_path)
    input_image = Image.open("/home/trgtuan/OneDrive/My Git/ComfyUI-OOTDiffusion-MaskOnly/model7.jpeg")  
    input_image = input_image.resize((768, 1024))
    parsed_image, face_mask = parsing(input_image.resize((384, 512)))
    
    parsed_image.save("/home/trgtuan/OneDrive/My Git/ComfyUI-OOTDiffusion-MaskOnly/parsed_image_model7.png")
    # img = np.array(parsed_image)
    # np.savetxt("parsed_image.txt", img, fmt="%d")