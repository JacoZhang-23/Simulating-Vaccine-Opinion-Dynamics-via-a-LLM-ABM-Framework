
import pickle

with open('data/input/user_embedding.pickle', 'rb') as f:
    data = pickle.load(f)

# 打印前几个 embedding
for i, embedding in enumerate(data[:5]):
    print(f"User {i} embedding shape: {embedding.shape}")
    print(embedding)