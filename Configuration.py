class Configuration:
    learning_rate = 0.001
    training_iters = 1000000
    batch_size = 32
    display_step = 1
    dropout = 0.5
    test_number = 5000  # number of test images (out of 250000)

    CHECKPOINT_PATH = './'
    TRAIN_FOLDER = './train128'
    SUBMISSION_FOLDER = './test128'
    RESULT_FILE_PATH = './results.dat'

    def __init__(self):
        pass
