from sklearn.metrics import mutual_info_score
import numpy as np
import random
import itertools


ans = [[i for _ in range(5)] for i in range(1000)]
ans = list(itertools.chain(*ans))
pred = random.sample(ans, len(ans))

print(mutual_info_score(ans, ans))
print(mutual_info_score(ans, pred))
