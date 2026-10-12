import json
from collections import Counter

def count_unsafe_turns(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    turn_counts = Counter()

    for item in data:
        turns = item.get("result", [])
        first_unsafe_turn = None

        for entry in turns:
            if entry.get("safety_judge", "").strip().lower() == "unsafe":
                first_unsafe_turn = entry.get("turn")
                break

        if first_unsafe_turn is not None and 1 <= first_unsafe_turn <= 5:
            turn_counts[first_unsafe_turn] += 1

    # Ensure all turns 1–5 are present (even if 0)
    result = {turn: turn_counts.get(turn, 0) for turn in range(1, 6)}

    return result


if __name__ == "__main__":
    file1 = "sizes_results/cognitive_manipulation_results32b_abliterated.json"
    file2 = "sizes_results/emotional_exploitation_results32b_abliterated.json"
    file3 = "sizes_results/persistence_&_subversion_results32b_abliterated.json"
    file4 = "sizes_results/social_role_framing_results32b_abliterated.json"

    counts1 = count_unsafe_turns(file1)
    counts2 = count_unsafe_turns(file2)
    counts3 = count_unsafe_turns(file3)
    counts4 = count_unsafe_turns(file4)
    print(counts1)
    print(counts2)
    print(counts3)
    print(counts4)

    total_counts = Counter(counts1) + Counter(counts2) + Counter(counts3) + Counter(counts4)
    print(total_counts)