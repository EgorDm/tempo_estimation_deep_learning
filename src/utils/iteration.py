class Iterator:
    def __init__(self, A, index=0, next_callback=None):
        self.index = index - 1
        self.A = A
        self.next_callback = next_callback
        self.next()

    def next(self, skip_callback=False):
        if not self.has_next(): return None
        self.index += 1
        if not skip_callback and self.next_callback is not None: self.next_callback(self, self.A[self.index])
        return self.A[self.index]

    def has_next(self):
        return self.index + 1 < len(self.A)

    def current(self):
        return self.A[self.index]

    def peek(self):
        if not self.has_next(): return None
        return self.A[self.index + 1]
