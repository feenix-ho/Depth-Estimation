import configparser

config = configparser.ConfigParser()
config.read('train_arg.txt')
data_path = config['DEFAULT']['data_path']  # drive/MyDrive/dataset
image_height = int(config['DEFAULT']['image_height'])  # 480
image_width = int(config['DEFAULT']['image_width'])  # 640
patch_size = int(config['DEFAULT']['patch_size'])  # 32
knowledge_dims = list(
    map(int, config['DEFAULT']['knowledge_dims'].split(',')))  # 4096, 2048, 1024
dense_dims = list(
    map(int, config['DEFAULT']['dense_dims'].split(',')))  # 1024, 1024, 1024, 1024
latent_dims = int(config['DEFAULT']['latent_dims'])  # 256
emb_size = int(config['DEFAULT']['emb_size'])  # 4096
readout = config['DEFAULT']['readout']  # ignore
hooks = list(
    map(int, config['DEFAULT']['hooks'].split(',')))  # 3, 6, 9, 12
batch_size = int(config['DEFAULT']['batch_size'])  # 4
num_epochs = int(config['DEFAULT']['num_epochs'])  # 50
learning_rate = float(config['DEFAULT']['learning_rate'])  # 1e-4
weight_decay = float(config['DEFAULT']['weight_decay'])  # 1e-2
adam_eps = float(config['DEFAULT']['adam_eps'])  # 1e-3


class Arg_train:
