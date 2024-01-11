import time


class StopWatch:
    def __init__(self):
        self.times = []
        self.job_names = []
        self.start_time = None
        self.job_name = None

    def start(self, name=''):
        self.start_time = time.time()
        self.job_name = name

    def stop(self):
        self.times.append(time.time() - self.start_time)
        self.job_names.append(self.job_name)
        return self.times[-1]

    def avg(self):
        return self.sum() / len(self.times)

    def sum(self):
        return sum(self.times)

    def display(self):
        print(f'{"job_name":<20} {"time(s)":>10}')
        print('-' * 40)
        for i in range(0, len(self.times)):
            print(f'{self.job_names[i]:<20} {self.times[i]:>10.2f}')
