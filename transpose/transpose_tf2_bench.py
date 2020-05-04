import tensorflow as tf
import time

repeats = 100

def bench_func(x, repeats=100):
  for i in range(repeats):
    y = tf.transpose(x)
  z = tf.math.reduce_sum(y)
  _ = z.numpy()
  return y

exp = 28
size = pow(2, exp)
dtypes = [tf.float16, ]
x = tf.ones(size, dtypes[0])
bench_func(x, 2) # warmup

for m in range(1, exp):
  for dtype in dtypes:
    dtype_str = "float16" if dtype == tf.float16 else "float32"
    for i in range(0, exp - m + 1):
      dim0 = pow(2, i)
      dim1 = pow(2, m)
      dim2 = pow(2, exp - i - m)
      x = tf.ones((dim0, dim1, dim2), dtype)
      start = time.time()
      bench_func(x, repeats)
      end = time.time()
      time_in_ms = (end - start) / repeats * 1000
      print("dtype: {} dim0: {} dim1: {} dim2: {} time(ms): {}"
                .format(dtype_str, dim0, dim1, dim2, time_in_ms))


