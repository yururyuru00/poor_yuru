import numpy as np

dim_idobjct_val = np.genfromtxt("./test.csv", delimiter=',' ,skip_header=1, dtype=(int, int, float))
print(dim_idobjct_val)