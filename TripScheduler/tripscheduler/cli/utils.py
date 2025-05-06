import json
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Tuple

def load_test_case(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_valid_combinations(
    places: List[Dict[str, Any]],
    windows: List[Tuple[int,int,Any]]
) -> List[Tuple[int, ...]]:
    """
    restaurant 카테고리별로 meal_type을 그룹핑하고
    itertools.product로 조합 생성 후 org_id 중복 제거
    """
    meal_groups: Dict[Any, List[int]] = defaultdict(list)
    for idx, place in enumerate(places):
        if place.get("category") == "restaurant":
            meal_type = windows[idx][2]
            if meal_type:
                meal_groups[meal_type].append(idx)

    raw_selections = list(itertools.product(*meal_groups.values())) if meal_groups else [()]

    valid: List[Tuple[int,...]] = []
    for sel in raw_selections:
        seen = set()
        ok = True
        for i in sel:
            org = places[i].get("org_id", places[i]["id"])
            if org in seen:
                ok = False
                break
            seen.add(org)
        if ok:
            valid.append(sel)
    return valid

def build_selection_inputs(
    places: List[Dict[str, Any]],
    windows: List[Tuple[int,int,Any]],
    sel: Tuple[int, ...]
) -> Tuple[List[Dict[str,Any]], List[Tuple[int,int,Any]], Tuple[Any,...]]:
    selected = set(sel)
    for i, p in enumerate(places):
        if p.get("category") != "restaurant":
            selected.add(i)

    selected = sorted(selected)
    sel_places  = [places[i]  for i in selected]
    sel_windows = [windows[i] for i in selected]
    labels = tuple(windows[i][2] for i in sel)
    return sel_places, sel_windows, labels

def display_results(
    results: Dict[Tuple[int,...], Any],
    windows: List[Tuple[int,int,Any]]
):
    for sel, out in results.items():
        labels = tuple(windows[i][2] for i in sel)
        print(f"\n=== Option {labels!r} ===")
        if not out or out.get("visits") is None:
            print("  (해결 불가)")
            continue

        print(f"  Total cost: {out['cost']}")
        for step in out["visits"]:
            print("   ", step)

        # path = out.get("path", [])
        # if path:
        #     print("  Full path coordinates:")
        #     for coord in path:
        #         print("    ", coord)
        # else:
        #     print("  (경로 정보 없음)")
