import numpy as np
import random

def add_blinking(input_exp):
    eye_close = np.array([
        0, 0, 0, 0, 1, -1, -1,
        0, -2, 1, 0, 0, 0,
        0, 0, 0, 0, -5, 0,
        0, 5, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0], dtype=np.float32)

    blink_start = 0

    scale = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0]

    while blink_start <= (input_exp.shape[0] - 30):
        blink_start += random.randint(60, 150)
        if blink_start >= (input_exp.shape[0]-30):
            break
        for j in range(12):
            scaled_transformation = eye_close * scale[j]
            # TODO: calculate steps as interpolation values between origin and blink tensor
            input_exp[blink_start+j] = np.clip((input_exp[blink_start+j] + scaled_transformation), -5, 5)   
    return input_exp