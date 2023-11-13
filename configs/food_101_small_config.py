config = {
    'path'      : '/home/usr/food-101/small_images',
    'height'    : 128,
    'width'     : 128,
    'train_size': 0.7,
    'eval_size' : 0.3,
    'batch_size': 16,
    'channel'   : 3,
    'horizontal_flip_rate': 0.5,
    'grayscale_rate': 0.05,
}

def load_config_food_101_small():

    return config