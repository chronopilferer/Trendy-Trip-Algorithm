import json

def explore_data(data, level=0, prev_value=None, max_repeat=3):
    indent = ' ' * (level * 2)  # 들여쓰기 처리
    if isinstance(data, list):
        print(f"{indent}List with {len(data)} items")
        if len(data) > max_repeat:
            print(f"{indent}Showing first {max_repeat} items:")
            for i, item in enumerate(data[:max_repeat]):
                print(f"{indent}  Item {i+1}:")
                explore_data(item, level + 1)
            print(f"{indent}...and {len(data) - max_repeat} more items")
        else:
            for i, item in enumerate(data):
                print(f"{indent}  Item {i+1}:")
                explore_data(item, level + 1)

    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}Key: {key}")
            if value is None:
                print(f"{indent}  Value: None")
            else:
                explore_data(value, level + 1, prev_value=value)

    else:
        # 값이 null 또는 None인 경우와 다른 값을 처리
        if prev_value == data:
            print(f"{indent}  Repeated value: {data}")
        else:
            print(f"{indent}  Value: {data}")

if __name__ == '__main__':
    with open('./directions_raw_data.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)

    print("Data structure:")
    explore_data(raw)  # 데이터를 탐색
