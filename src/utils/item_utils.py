"""Helpers for item dictionaries and legacy tuple support."""


def get_item_dims(item):
    if isinstance(item, dict):
        l = item.get('l', item.get('length'))
        w = item.get('w', item.get('width'))
        h = item.get('h', item.get('height'))
    else:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError("Item must be dict or tuple/list with 3 dimensions")
        l, w, h = item[0], item[1], item[2]
    if l is None or w is None or h is None:
        raise ValueError("Item dimensions must be set")
    return int(l), int(w), int(h)


def get_item_stacking(item, default='stackable'):
    if isinstance(item, dict):
        return item.get('stacking', default)
    return default


def make_item(l, w, h, stacking='stackable'):
    return {
        'l': int(l),
        'w': int(w),
        'h': int(h),
        'stacking': stacking,
    }
