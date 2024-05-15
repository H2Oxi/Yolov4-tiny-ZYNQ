
def quantize_float_to_fixed_point(float_value, num_bits):  
    """  
    Quantize a float to a fixed-point number with a given number of bits.  
      
    :param float_value: The float value to quantize.  
    :param num_bits: The number of bits to use for the fixed-point representation.  
    :return: The quantized fixed-point value and the quantization error.  
    """  
    # Check if the number of bits is valid  
    if num_bits < 1 or num_bits > 64:  
        raise ValueError("Number of bits must be between 1 and 64.")  
      
    # Calculate the scale factor (2^(num_bits-1) for signed integers)  
    scale_factor = 2 ** (num_bits - 1)  
      
    # Quantize the float value to the nearest integer  
    quantized_value = round(float_value * scale_factor)  
      
    # Calculate the quantization error  
    quantization_error = abs(float_value - quantized_value / scale_factor)  
      
    # Convert the quantized value back to a float for easier comparison  
    quantized_value_r = quantized_value / scale_factor  
      
    return quantized_value_r, quantization_error , quantized_value
  
if __name__ == "__main__":  
    # Example usage  
    float_value = 3.14159  
    num_bits_list = [8, 16, 32]  
    
    for num_bits in num_bits_list:  
        quantized_value_r, quantization_error ,quantized_value_q= quantize_float_to_fixed_point(float_value, num_bits)  
        print(f"Quantized to {num_bits} bits: {quantized_value_r:.4f}, Quantization Error: {quantization_error:.6f},Quantized to {quantized_value_q}")