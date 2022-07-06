import numpy as np
import torch
import torch.nn as nn
import sys



brain = np.random.uniform(0, 1, (4, 360, 32))
hidden = np.random.normal(0, 1, (4, 1, 512))

# want [4, 360, 544]

hidden = np.repeat(hidden, 360, axis=1)
context = np.concatenate((brain, hidden), axis=-1)

print("brain:", brain.shape)
print("hidden:", hidden.shape)
print("context:", context.shape)

mh = nn.MultiheadAttention(544, 8, batch_first=True)


context = context.astype(np.float32)
context = torch.from_numpy(context)
print("context:", context.shape)
print()

attn_output, attn_output_weights = mh(context, context, context)
print("attn_output:", attn_output.shape)
print("attn_output_weights:", attn_output_weights.shape)  # [bs, 360, 360]
print()


y = np.mean(attn_output_weights.detach().numpy(), axis=1)
#y = attn_output_weights.detach().numpy()[0]
y = np.expand_dims(y, axis=-1)
print("y:", y.shape)
print(sum(y[0]))
print()

print(y[0, :10])
print()

z = torch.sum(context * y, dim=1)
print("z:", z.shape)


sys.exit(0)
print("---------- Multihead two ---------------")

brain = np.random.uniform(0, 1, (4, 360, 32))
hidden = np.random.normal(0, 1, (4, 1, 32))#512))
hidden = np.repeat(hidden, 360, axis=1)
print("brain:", brain.shape)
print("hidden:", hidden.shape)

mh = nn.MultiheadAttention(32, 8, batch_first=True)

attn_output, attn_output_weights = mh(brain, hidden, brain)
print("attn_output:", attn_output.shape)
print("attn_output_weights:", attn_output_weights.shape)
print()

