import heapq
last_name = "huang"
last_name = "aabbbccdddd"

def make_fq(input):
  sa = []
  ps = {}
  total =
  for c in input:
    if c in ps:
      ps[c] = ps[c] + 1
    else:
      ps[c] = 1
  return ps


# Use heapify to rearrange the elements
fq = make_fq(last_name)
tl = []

for k in fq:
  tl.append((k,fq[k]))
print(tl)
def getKey(item):
  return item[1]

tl = sorted(tl, key=getKey)

class Node:
  def __index__(self, name, freq):
    self.name = name
    self.freq = freq


for i in range(1, len(tl),2):
  item1 = tl[i]
  item2 = tl[i+1]
  name = item1[0] + item2[0]
  freq = item1[1] + item2[1]
  node = Node(name,freq)


class MinHeap:
  

  

class Symbol:
  def __init__(self, sym):
    self.sym = sym
  
  def myfunc(self):
    print("hello "+self.sym)
  

