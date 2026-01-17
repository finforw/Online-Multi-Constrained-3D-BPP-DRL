from data.cutter import Cutter
import torch

cutter = Cutter(10, 10, 10)
res = []
for _ in range(2100):
    boxes = cutter.generate_boxes()
    truncated_data = [sublist[:3].tolist() for sublist in boxes]
    res.append(truncated_data)
    cutter.reset()

print(res)
torch.save(res, 'test_data/fizz_fuzz.pt')
