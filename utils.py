import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'

def build_dataset(config):

    def load_dataset(path, head_size=90, tail_size=220):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                head_tokens = token[:head_size]
                tail_tokens = token[-tail_size:]
                token = head_tokens + tail_tokens
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if head_size + tail_size > seq_len:
                    mask = [1] * len(token_ids) + [0] * (head_size + tail_size - seq_len)
                    token_ids += ([0] * (head_size + tail_size - seq_len))
                else:
                    mask = [1] * (head_size + tail_size)
                    token_ids = token_ids[:head_size] + token_ids[-tail_size:]
                    seq_len = head_size + tail_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.head_size, config.tail_size)
    dev = load_dataset(config.dev_path, config.head_size, config.tail_size)
    test = load_dataset(config.test_path, config.head_size, config.tail_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, head_size, tail_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.head_size = head_size
        self.tail_size = tail_size

    def _to_tensor(self, datas):
        x = []
        for data in datas:
            tokens = data[0]
            seq_len = len(tokens)
            if seq_len > self.head_size + self.tail_size:
                selected_tokens = tokens[:self.head_size] + tokens[-self.tail_size:]
            else:
                selected_tokens = tokens + [0] * (self.head_size + self.tail_size - seq_len)
            x.append(selected_tokens)
        x = torch.LongTensor(x).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.head_size, config.tail_size)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



