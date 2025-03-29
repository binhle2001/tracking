def push(queue, key, value, max_size):
    if len(queue) > max_size:
        # Xóa key đầu tiên (key cũ nhất)
        oldest_key = next(iter(queue))
        del queue[oldest_key]
    queue[key] = value

def pop(queue):
    if queue:
        # Lấy key và giá trị đầu tiên
        oldest_key = next(iter(queue))
        value = queue[oldest_key]
        del queue[oldest_key]
        return oldest_key, value
    else:
        return None, None 