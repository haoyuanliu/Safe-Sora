import torch

def extract(image, avg = 0):
    image = image.view(-1, 256, 256)
    image = image.mean(dim=0)
    extract_watermark = torch.zeros(256, dtype=torch.float32, device=image.device)   
    for index in range(256):
        key = image[index//16*16:(index//16*16 + 16), index*16%256:(index*16%256 + 16)].sum()
        key = key / 256.0
        if key > avg:
            value = 1.0
            extract_watermark[index] = value
        else:
            value = 0.0
            extract_watermark[index] = value
    return extract_watermark, bits_to_hex(extract_watermark)

def extract_with_conv(image, model, avg = 0):
    input_tensor = image.mean(dim=1, keepdim=True)  
    output_tensor = model(input_tensor)
    output_tensor = output_tensor.view(-1)
    output_tensor = (output_tensor > avg).float()
    return output_tensor, bits_to_hex(output_tensor)   

def bits_to_hex(bits):
    str_watermark = ''
    for i in range(len(bits)):
        bits[i] = int(bits[i].item())
        if bits[i] < 1.0:
            str_watermark += '0'
        else:
            str_watermark += '1'
    str_watermark = hex(int(str_watermark, 2))[2:] 
    return str_watermark
