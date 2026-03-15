from data.cutter import Cutter
import torch

cutter = Cutter(10, 10, 10)
res = []
for _ in range(2100):
    boxes = cutter.generate_boxes()
    processed_data = [subarray.tolist() for subarray in boxes]
    res.append(processed_data)
    cutter.reset()

print(res)
torch.save(res, 'test_data/fizz_fuzz.pt')
