import math
import torch

class CNN_MLP(torch.nn.Module):
    """ CNN-MLP with 1 Conv layer, 1 Max Pool layer, and 1 Linear layer. """

    def __init__(self, seq_len=220, embed_size=64, vocab_size=45, pad_index=0, 
                 stride=1, kernel_size=3, conv_out_size=64, hidden_layer_sizes=[128, 64, 32, 8, 1], dropout_rate=0.25):    
        super(CNN_MLP, self).__init__()
        
        # Embedding layer parameters
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_out_size = conv_out_size
        
        # Misc
        self.dropout_rate = dropout_rate
        
        # Conv Layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_index)
        
        self.conv = torch.nn.Conv1d(self.seq_len, self.conv_out_size, self.kernel_size, self.stride)
        self.hidden_act = torch.relu
        self.max_pool = torch.nn.MaxPool1d(self.kernel_size, self.stride)        
        self.flatten = lambda x: x.view(x.shape[0], x.shape[1]*x.shape[2])
        
        # MLP layers
        self.fc_layers = []
        self.hidden_layer_sizes.insert(0, self._linear_layer_in_size())
        for i in range(len(self.hidden_layer_sizes) - 1):
            self.fc_layers.append(torch.nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            self.fc_layers.append(torch.nn.ReLU())
            if self.dropout_rate and i != len(self.hidden_layer_sizes) - 2:
                self.fc_layers.append(torch.nn.Dropout(self.dropout_rate))
        self.fc_layers.append(torch.sigmoid)
        
    def _linear_layer_in_size(self):
        out_conv_1 = ((self.embed_size - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
                            
        return out_pool_1*self.conv_out_size
    
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.conv(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        return x.squeeze()

    def embed(self, x):
        x = self.embedding(x)
        
        x = self.conv(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)

        x = self.flatten(x)

        for i, layer in enumerate(self.fc_layers):
            if i != len(self.fc_layers) - 1:
                x = layer(x)
        
        return x
