class Configuration:
    learning_rate = 0.0005
    training_iters = 1000000
    batch_size = 128
    display_step = 1
    dropout = 0.5
    test_number = 5000  # number of test images (out of 250000)

    CHECKPOINT_PATH = './1'
    TRAIN_FOLDER = '../internet/train64'
    SUBMISSION_FOLDER = '../test64'
    RESULT_FILE_PATH = './1/results.dat'

    def __init__(self):
        pass
