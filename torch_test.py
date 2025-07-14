import torch
print(torch.__version__)  # Should show something like 2.7.1+cu121
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should print your GPU name