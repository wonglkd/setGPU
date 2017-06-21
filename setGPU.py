import os
import gpustat
import random


# https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"


stats = [gpu for gpu in gpustat.GPUStatCollection.new_query() if gpu.entry['uuid'] is not None]
ids = map(lambda gpu: int(gpu.entry['index']), stats)
ratios = map(lambda gpu: 1. - float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
ratios = [r ** 5. for r in ratios]  # Favour free ones even more strongly
bestGPU = weighted_choice(zip(ids, ratios))

print("setGPU: Setting GPU to: {}".format(bestGPU))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
