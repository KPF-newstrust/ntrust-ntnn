import math


def find_first(fn, arr):
    for item in arr:
        if fn(item): return item
    return None

def arr_from(arr, size, padding):
    if len(arr) >= size: 
        return arr[:size]
    return arr + ([padding] * (size - len(arr)))


def kfold(x, y, partitions=5):
    def to_partition(coll):
        sz = math.ceil(len(coll) / partitions)
        return [coll[sz*i: sz*(i+1)] for i in range(partitions)]
    x_parts = to_partition(x)
    y_parts = to_partition(y)

    for test_i in range(partitions):
        yield ([j for i in range(len(x_parts)) for j in x_parts[i] if i != test_i],
            [j for i in range(len(y_parts)) for j in y_parts[i] if i != test_i],
            x_parts[test_i],
            y_parts[test_i])


def split_arr(arr, nitems):
    for i in range(0, len(arr), nitems):
        yield arr[i:i+nitems]
