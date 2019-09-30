import time


def start_timer():
    return time.time()


def elapsed_time_seconds(timer_):
    return time.time() - timer_


def elapsed_time_ms(timer_):
    return elapsed_time_seconds(timer_) * 1000


def elapsed_time_minutes(timer_):
    return elapsed_time_seconds(timer_) / 60.0


if __name__ == '__main__':
    timer = start_timer()
    time.sleep(1.0)
    print(elapsed_time_ms(timer))
    print(elapsed_time_seconds(timer))
