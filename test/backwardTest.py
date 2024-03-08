import torch

torch.random.manual_seed(1)

a = torch.randn([30, 8, 768], dtype=torch.float32)

linear = torch.randn([30, 768, 1], dtype=torch.float32, requires_grad=True)

score = torch.matmul(a, linear)

# score = torch.randn([30,8], dtype=torch.float32, requires_grad=True)
g = torch.randn([30,8,1], dtype=torch.float32)
score.backward(g)
# _, idx = torch.max(score, dim=-1)



# new_idx = idx * 8

# mask = torch.zeros([30])

# mask[0] = 1 # 只有0错了

# loss = _.sum()

# loss.backward()

print(123)
