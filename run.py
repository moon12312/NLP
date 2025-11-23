import time
import torch
import numpy as np
from train_eval import train, test,evaluate,predict,init_network
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
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    # test
    test(config, model, test_iter)

    # evaluate
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    print('Test Accuracy:', test_acc)
    print('Test Loss:', test_loss)
    print('Test Report:', test_report)
    print('Test Confusion Matrix:', test_confusion)

    # predict
    pred_iter = build_iterator(test_data, config)
    predictions = predict(config, model, pred_iter)
    print('Predictions:', predictions)