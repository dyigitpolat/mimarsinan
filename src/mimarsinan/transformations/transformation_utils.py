import torch

def transform_np_array(weight_array, transformation):
    weight_tensor = torch.from_numpy(weight_array)
    quantized_weight_tensor = transformation(weight_tensor)
    return quantized_weight_tensor.detach().numpy()
