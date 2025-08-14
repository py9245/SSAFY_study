from collections import deque

# class Queue:
#     def __init__(self, capacity=10):
#         self.capacity = capacity
#         self.items = [None] * capacity
#         self.front = -1
#         self.rear = -1
#
#     def is_full(self):
#         return self.rear == self.capacity - 1
#
#     def enqueue(self, item):
#         if self.is_full():
#             raise IndexError('Queue is Full')
#         self.rear += 1
#         self.items[self.rear] = item
#
#     def is_empty(self):
#         return self.front == self.rear
#
#     def dequeue(self):
#         if self.is_empty():
#             raise IndexError('Queue is Empty')
#         self.front += 1
#         item = self.items[self.front]
#         self.items[self.front] = None
#         return item
#
#     def peek(self):
#         if self.is_empty():
#             raise IndexError('Queue is Empty')
#         return self.items[self.front + 1]

# queue = Queue()


# class Queue:
#     def __init__(self, capacity=10):
#         self.capacity = capacity + 1
#         self.items = [None] * self.capacity
#         self.front = 0
#         self.rear = 0
#
#     def is_full(self):
#         return (self.rear + 1) % self.capacity == self.front
#
#     def enqueue(self, item):
#         if self.is_full():
#             raise IndexError('Queue is Full')
#         self.rear = (self.rear + 1) % self.capacity
#         self.items[self.rear] = item
#
#     def is_empty(self):
#         return self.front == self.rear
#
#     def dequeue(self):
#         if self.is_empty():
#             raise IndexError('Queue is Empty')
#         self.front = (self.front + 1) % self.capacity
#         item = self.items[self.front]
#         self.items[self.front] = None
#         return item
#
#     def peek(self):
#         if self.is_empty():
#             raise IndexError('Queue is Empty')
#         return self.items[(self.front + 1) % self.capacity]
#
#
# queue = Queue(10)
# for i in range(30):
#     queue.enqueue(i)
#     if queue.is_full():
#         print(f"is_Full popleft 한 itmes : {queue.items}")
#         queue.dequeue()
#     print(queue.items)


T = int(input())

for case in range(1, T + 1):
    N = int(input())
    last_p = 1
    best_p = 1
    q = deque([[1, 1]])
    # 1번이 대기열에 들어감 처음 오니 캔디 1개 받을 수 있음

    # 캔디 동나면 끝
    while N > 0:

        # 대기열에 맨 앞에 있는 사람의 번호p 받을 수 있는 캔디c
        p, c = q.popleft()

        # 방금 받아놓고 또 받으려고 대기열 마지막에 달려감 근데 방금 받았던거보다 하나 더 받을 수 있음
        q.append([p, c + 1])

        # 방금 받은 사람의 번호를 last_p에 갱신 하고 c만큼 캔디를 줌
        last_p = p
        N -= c

        # 딱 맞게 가지가서 남은캔디가 0이든 모자라서 덜가지가든 남은 캔디가 없으니 종료
        if N < 1:
            break

        best_p += 1
        q.append([best_p, 1])

    print(last_p)