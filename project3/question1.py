from math import log2


class Empty(Exception):
    pass


# standard heap implementation in array form
class BaseHeapQueue:
    class Item:
        # initialized with key value pair
        # which is passed with frequency and the alphabet
        def __init__(self, k, v):
            self.key = k
            self.val = v

        def __lt__(self, other):
            """compare first with key, i.e, the frequency/prob,
             if tight, then compare with value, i.e. the alphabet"""
            if self.key != other.key:
                return self.key < other.key
            return self.val < other.val


        def __repr__(self):
            return "({0}: {1})".format(self.key, self.val)


class HeapPriorityQueue(BaseHeapQueue):

    def __init__(self):
        self._data = []
    # find node parent
    def parent(self, j):
        return (j - 1) // 2
    #find left child
    def left(self, j):
        return j * 2 + 1
    #find right child
    def right(self, j):
        return 2 * j + 2

    def has_left_child(self, j):
        return self.left(j) < len(self._data)

    def has_right_child(self, j):
        return self.right(j) < len(self._data)

    def is_empty(self):
        return len(self) == 0

    def swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
    #if the node is smaller than its parent, we move it up
    def upheap(self, j):
        parent = self.parent(j)
        if j > 0 and self._data[j] < self._data[parent]:  # by default a min heap if key is not negated
            self.swap(j, parent)
            self.upheap(parent)
    # if node has bigger than its children, we move it down by swapping the smaller of the child with the node
    def downheap(self, j):
        if self.has_left_child(j):
            left = self.left(j)
            small = left
            if self.has_right_child(j):
                right = self.right(j)
                if self._data[left] > self._data[right]:
                    small = right
            if self._data[j] > self._data[small]:
                self.swap(j, small)
                self.downheap(small)

    def __len__(self):
        return len(self._data)
    # add a node, always insert in the last position then upheap
    def add(self, key, val):
        item = self.Item(key, val)
        self._data.append(item)
        self.upheap(len(self._data) - 1)
        return item
    # get the min element
    def min(self):
        if self.is_empty():
            raise Empty("queue is empty")
        item = self._data[0]
        return (item.key, item.val)
    # remove the min element
    def remove_min(self):
        if self.is_empty():
            raise Empty("queue is empty")
        self.swap(len(self._data) - 1, 0)
        item = self._data.pop()
        self.downheap(0)
        return item
    #return the minimum item
    def peek(self):
        if self.is_empty():
            raise Empty("queue is empty")
        item = self._data[0]
        return item


class Huffman:
    def __init__(self, q):
        self.q = q
        self.arr = []
    #compression, using the established heap,
    #remove two minimum nodes at a time, and join two nodes with their keys added, and insert back,
    # the values are also joined to distinguish from end node, the new node also keeps reference to the deleted nodes.
    # so in the end, we transform the heap into a binary tree
    def compress(self):
        while len(self.q) > 1:
            item1 = self.q.remove_min()
            item2 = self.q.remove_min()
            k = item1.key + item2.key
            v = item1.val + item2.val
            item = self.q.add(k, v)
            item.left = item1
            item.right = item2
    #if it is symbol, the value should 1 length long
    def is_symbol(self, str):
        return len(str) == 1

    # breadth first search to traverse the huffman tree,
    # the left branch is assigned 1 and right branch assigned with 0
    # return a list of tuple, where first element is the symbol, the second element is the code bit
    def bfs(self):
        arr = []
        temp = HeapPriorityQueue()
        k = 0
        item = self.q.peek()
        item.code = ""
        temp.add(k, item)
        while len(temp):
            item = temp.remove_min()
            item = item.val
            val, code = item.val, item.code
            if self.is_symbol(val):
                arr.append((val, code))
            if hasattr(item, 'left'):
                k = k + 1
                left = item.left
                left.code = code + "0"
                temp.add(k, left)
            if hasattr(item, 'right'):  # huffman tree should have both left and right child at the same time, but here I check anyway
                k = k + 1
                right = item.right
                right.code = code + "1"
                temp.add(k, right)
        return arr

    def show_code(self):
        arr = []
        arr = self.bfs()
        for item in arr:
            print("({0},{1})\t".format(item[0], item[1]))


# make frequency table
def make_fq(kv):
    ps = {}
    for c in kv:
        if c in ps:
            ps[c] = ps[c] + 1
        else:
            ps[c] = 1
    return ps

#transform from tuple to dictionary
def from_tuple_to_dict(arr):
    result = {}
    for item in arr:
        result[item[0]] = item[1]
    return result


# inverse the key value pair in a dict
# assume value is also unique
def inverse_dict(kv):
    vk = {}
    for k, v in kv.items():
        vk[v] = k
    return vk


# calculate the entropy
def calc_entropy(pr):
    ent = 0
    for k, p in pr.items():
        ent += - p * log2(p)
    return ent


# calculate probability table
def make_prob(fq):
    total = sum(v for k, v in fq.items())
    return {k: v/total for k, v in fq.items()}


# calculate weighted path
def calc_weighted_path(pr, result):
    #pr: probability table
    #result: encoding result
    wp = 0
    for k, p in pr.items():
        wp += p * len(result[k])
    return wp


def print_encoding_in_order(encoding):
    for item in encoding:
        k = item[0]
        v = item[1]
        print("{}\t {}".format(k, v))

# test if the total probs sum to 1 with each individual component 2^(-|code|)
def test1(result):
    total_prob = 0
    for k, v in result.items():
        total_prob += 2 ** (-len(v))
    print("total probability: {}".format(total_prob))


def decode(str, vk):
    output = []
    key = ""
    for i in range(len(str)):
        key = key + str[i]
        if key in vk:
            output.append(vk[key])
            key = ""
    return "".join(output)


if __name__ == "__main__":
    sample = "Huang"
    print("the string to be encoded is: {}".format(sample))
    q = HeapPriorityQueue()
    fq = make_fq(sample)
    pr = make_prob(fq)

    for k, v in fq.items():
        q.add(v, k)

    # while not q.is_empty():
    #     print(q.remove_min())

    h = Huffman(q)
    h.compress()
    encoding = h.bfs()

    encoding = sorted(encoding, key = lambda item: (len(item[-1]),item))

    encoding_dict = from_tuple_to_dict(encoding)
    kv = inverse_dict(encoding_dict)
    test1(encoding_dict)
    ent = calc_entropy(pr)
    wp = calc_weighted_path(pr, encoding_dict)

    # print('frequency table:')
    # print(fq)
    print('encoding in order by code bit length and alphabet:')
    print_encoding_in_order(encoding)
    output = []
    for c in sample:
        output.append(encoding_dict[c])
    encoded_string = "".join(map(str, output))
    print("The encoded string for the name is: {}".format(encoded_string))
    print("decoded string is: {}".format(decode(encoded_string, kv)))
    print("entropy is : {}".format(ent))
    print("weighted path is : {}".format(wp))
