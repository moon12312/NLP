import time
import torch
import numpy as np
from train_eval import evaluate, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Data'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    _, _, test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load('modell.pth'))  # Replace 'your_pretrained_model.pth' with the path to your pretrained model file


    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    print('Test Accuracy:', test_acc)
    print('Test Loss:', test_loss)
    print('Test Report:', test_report)
    print('Test Confusion Matrix:', test_confusion)
