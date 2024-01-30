class Node(object):
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node


class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, value):
        node = Node(value)
        if not self.head:
            self.head = node
            self.tail = self.head
        else:
            self.tail.next_node = node
            self.tail = node
        self.size += 1

    def front(self):
        if not self.head:
            return Node
        return self.head.data

    def pop(self):
        if self.head:
            self.head = self.head.next_node
            self.size -= 1

    def __iter__(self):
        self.current = self.head
        return self

    def __next__(self):
        if self.current:
            value = self.current.data
            self.current = self.current.next_node
            return value
        else:
            raise StopIteration

    def __len__(self):
        return self.size
