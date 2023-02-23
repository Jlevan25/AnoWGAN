import time


class Timer:
    def __init__(self, timer_name=''):
        self.start_time = 0
        self.timer_name = timer_name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        print(f'{self.timer_name} time {time.perf_counter() - self.start_time:.4} sec')


if __name__ == '__main__':
    lis_t = []
    list_ = []
    len_ = 100_000
    with Timer('range'):
        for i in range(1, len_//5+1):
            for j in range(5):
                pass
            lis_t.append(i * 5)

    with Timer('if'):
        for i in range(1, len_+1):
            if i % 5 == 0:
                list_.append(i)

    print(list_ == lis_t)
