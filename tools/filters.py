

def exp_average(values, weight=0.8):
    filtered = []
    last = None

    for value in values:
        last = last * weight + value * (1 - weight) if last else value
        filtered.append(last)

    return filtered
