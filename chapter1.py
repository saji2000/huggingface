# Save as chapter1_gpu.py
from transformers import pipeline
import torch


translator = pipeline("translation", model="abbasmahmudiai/MT5_en_to_persian")
translation = translator("I love Canada")

print(translation[0])