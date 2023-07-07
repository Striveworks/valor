def create(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def read(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def update(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def delete(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def background(fn: callable) -> callable:
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper