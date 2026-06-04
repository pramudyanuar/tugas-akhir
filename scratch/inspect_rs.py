import torch

dataset_path = "dataset/rs.pt"
data = torch.load(dataset_path, map_location="cpu")

print("Type:", type(data))

if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    for k, v in data.items():
        print(f"Key: {k}, Type: {type(v)}")
        if hasattr(v, 'shape'):
            print(f"  Shape: {v.shape}")
        elif hasattr(v, '__len__'):
            print(f"  Length: {len(v)}")
            if len(v) > 0:
                print(f"  First element type: {type(v[0])}")
                print(f"  First element: {v[0]}")
elif isinstance(data, list):
    print("Length:", len(data))
    print("First item type:", type(data[0]))
    print("First item:", data[0])
elif isinstance(data, tuple):
    print("Tuple length:", len(data))
    print("First element type:", type(data[0]))
    print("First element:", data[0])
else:
    print(data)
