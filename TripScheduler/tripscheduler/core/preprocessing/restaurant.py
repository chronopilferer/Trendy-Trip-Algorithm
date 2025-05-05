import logging

logger = logging.getLogger(__name__)

def split_restaurant_nodes(places, windows_map):
    new_places, new_wins = [], []

    logger.info("split_restaurant_nodes() 시작: 총 %d개 장소", len(places))

    for place in places:
        pid = place["id"]
        wins = windows_map.get(pid)

        if wins is None:
            logger.error("장소 %s의 유효 시간 윈도우가 없음", pid)
            raise ValueError(f"장소 {pid}의 유효 시간 윈도우가 없습니다.")

        if place.get("category") == "restaurant" and len(wins) > 1:
            logger.debug("레스토랑 분할: %s (%d개 윈도우)", place["name"], len(wins))
            for o, c, meal in wins:
                node = {**place}
                label = meal or "default"
                node.update({
                    "name": f"{place['name']} ({label})",
                    "id": f"{pid}_{label}",
                    "org_id": pid
                })
                new_places.append(node)
                new_wins.append((o, c, meal))
                logger.debug("  - 분할 노드 생성: %s [%d~%d]", node["name"], o, c)
        else:
            new_places.append(place)
            new_wins.append(wins[0] if wins else (None, None, None))
            logger.debug("레스토랑 아닌 장소 또는 단일 윈도우: %s", place["name"])

    logger.info("split_restaurant_nodes() 완료: 최종 %d개 장소", len(new_places))
    return new_places, new_wins
