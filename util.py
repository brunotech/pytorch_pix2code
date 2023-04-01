from torch.utils import data
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import pickle
import torch

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PLACEHOLDER = ' '
# CONTEXT_LENGTH = 48
image_size = 256


class Vocabulary:
    
    def __init__(self, file_path):
        self.load_vocab(file_path)
        self.length = len(self.vocab_to_index)
    
    def load_vocab(self, file_path):
        self.vocab_to_index = {}
        with open(file_path, 'rb') as vocab_file:
            self.vocab_to_index = pickle.load(vocab_file)
        self.index_to_vocab = {value:key for key, value in self.vocab_to_index.items()}
    
    def to_vec(self, word):
        vec = np.zeros(self.length)
        vec[self.vocab_to_index[word]] = 1
        return vec
       
    def to_vocab(self, index):
        return self.index_to_vocab[index]

class UIDataset(data.Dataset):
    
    def __init__(self, file_path, vocab_file_path):
        self.file_path = file_path
        self.paths = []
        self.get_paths()
        self.transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
        ])
        self.vocab = Vocabulary(vocab_file_path)
        
    def get_paths(self):
        for f in os.listdir(self.file_path):
            if f.find('.gui') != -1:
                file_name = f[:f.find('.gui')]
                if os.path.isfile(f'{self.file_path}/{file_name}.png'):
                    self.paths.append(file_name)
    
    def __len__(self):
        return(len(self.paths))
    
    def __getitem__(self, index):
        image = self.transform(
            Image.open(f'{self.file_path}/{self.paths[index]}.png')
        )[:-1]
        context, prediction = self.read_gui(
            f'{self.file_path}/{self.paths[index]}.gui'
        )
        return image, context, prediction
    
    def read_gui(self, file_path):
        context = []
        prediction = []

        # Tokenize the target code and ads start and end token
        token_sequence = [PLACEHOLDER, START_TOKEN]
        with open(file_path, 'r') as f:
            for line in f:
                line = line.replace(',', ' ,').replace('\n', ' \n')
                tokens = line.split(' ')
                token_sequence.extend(iter(tokens))
        token_sequence.append(END_TOKEN)

        # Generates cotext prediction pair
        context = token_sequence[:-1]
        prediction = token_sequence[1:]

        prediction_vec = [self.vocab.to_vec(word) for word in prediction]
        context_vec = [self.vocab.to_vec(word) for word in context]
        return torch.tensor(context_vec, dtype=torch.float), torch.tensor(prediction_vec, dtype=torch.float)