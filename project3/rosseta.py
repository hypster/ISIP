from heapq import heappush, heappop, heapify
from collections import defaultdict


def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

from huffman_encoding import *

if __name__ == '__main__':
    # txt = "this is an example for huffman encoding"
    txt = "Huang"
    # sample = [('A', 5), ('B', 25), ('C', 2.5), ('D', 12.5)]
    # symb2freq = from_tuple_to_dict(sample)

    symb2freq = defaultdict(int)
    for ch in txt:
        symb2freq[ch] += 1


    print(symb2freq)
    huff = encode(symb2freq)
    print ("Symbol\tWeight\tHuffman Code")
    for p in huff:
        print ("%s\t%s\t%s" % (p[0], symb2freq[p[0]], p[1]))

    pr = make_prob(symb2freq)
    ent = calc_entropy(pr)
    encoding = from_tuple_to_dict(huff)
    wp = calc_weighted_path(pr, encoding)

    print(symb2freq)
    print(encoding)
    print("entropy is : {}".format(ent))
    print("weighted path is : {}".format(wp))
