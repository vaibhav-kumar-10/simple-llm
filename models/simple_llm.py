import torch
import torch.nn as nn
import tiktoken

class SimpleLLM(nn.Module):

    vocab_size = 50257
    embedding_dim = 768
    expansion_factor = 4
    expansion_dim = embedding_dim * expansion_factor

    def __init__(self):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Define the weight layers HERE so they are saved/trained
        self.key_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.query_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attention_layer1 = nn.Linear(self.embedding_dim, self.expansion_dim)
        self.relu = nn.ReLU
        self.attention_layer2 = nn.Linear(self.expansion_dim, self.embedding_dim)

        self.decoder = nn.Linear(self.embedding_dim, self.vocab_size)

    def attention(self, x):
        # Placeholder for the attention mechanism
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)
        x = self.kvq_math(key, query, value)

        # TODO: Add mask

        # Run through attention MLP
        x = self.attention_layer1(x)
        x = self.relu(x)
        x = self.attention_layer2(x)

        return x
    
    def kvq_math(self, key, query, value):
        
        # Take cross product of key and query
        key_transpose = key.transpose(-2, -1) # TODO: Fix tranpose math here
        cross_attention = torch.matmul(query, key_transpose)
        cross_attention = torch.softmax(cross_attention)
        result = torch.multiply(value, cross_attention)

        return result

    def tokenizer(self, text):
        # Placeholder for the tokenizer
        # Splits text up into suface forms
        # Calls downloaded map to convert surface forms to tokens

        # Use tiktoken to tokenize the input text
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        x = torch.tensor(tokens).unsqueeze(0)
        return x
    
    def embed_tokens(self, tokens):
        # Placeholder for the embedding layer
        # Runs tokens through a single layer to get embeddings
        # Forward pass implementation should be simple linear transformation (nn.Embedding handles performance)
        return self.embedding(tokens)

    def forward(self, x):
        # Placeholder for the forward pass
        x = self.tokenizer(x)
        x = self.embed_tokens(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x