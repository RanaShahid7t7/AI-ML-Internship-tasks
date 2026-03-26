LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

def label_name(idx):
    return LABELS.get(int(idx), str(idx))
