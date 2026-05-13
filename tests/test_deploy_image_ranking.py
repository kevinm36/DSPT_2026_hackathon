"""
Test the two-step image ranking agent with ground-truth validation.

Picks random ad images from ADS16 category folders, classifies them via the agent,
compares classification against ground truth, then scores against a user profile.

Usage:
    python tests/test_deploy_image_ranking.py
    python tests/test_deploy_image_ranking.py --limit 5
    python tests/test_deploy_image_ranking.py --no-cleanup
"""

import argparse
import base64
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agentcore.deploy import deploy_agent, invoke_agent, delete_agent
from src.data_loader.profile_builder import build_user_profile

AGENT_NAME = "image_ranking_agent_test"
AGENT_SRC = Path(__file__).resolve().parents[1] / "image_ranking_agent_src/image_ranking_agent.py"

CORPUS_ROOTS = [
    Path("/Users/abhkumar55/Code/dev_code/dspt_creative_optimization_service/archive"
         "/ADS16_Benchmark_part1/ADS16_Benchmark_part1/Corpus/Corpus"),
    Path("/Users/abhkumar55/Code/dev_code/dspt_creative_optimization_service/archive"
         "/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Corpus/Corpus"),
]
TEST_USER_ID = "U0001"


def find_corpus_root(user_id: str) -> str:
    """Find which corpus root contains this user."""
    for root in CORPUS_ROOTS:
        if (root / user_id).is_dir():
            return str(root)
    raise FileNotFoundError(f"User {user_id} not found in any corpus root")

# Ads directories (folder number -> category)
ADS_ROOTS = [
    Path("/Users/abhkumar55/Code/dev_code/dspt_creative_optimization_service/archive"
         "/ADS16_Benchmark_part1/ADS16_Benchmark_part1/Ads/Ads"),
    Path("/Users/abhkumar55/Code/dev_code/dspt_creative_optimization_service/archive"
         "/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads"),
]

# Folder number (1-indexed) -> category name
FOLDER_TO_CATEGORY = {
    1: "Clothing & Shoes",
    2: "Automotive",
    3: "Baby Products",
    4: "Health & Beauty",
    5: "Media (BMVD)",
    6: "Consumer Electronics",
    7: "Console & Video Games",
    8: "DIY & Tools",
    9: "Garden & Outdoor living",
    10: "Grocery",
    11: "Kitchen & Home",
    12: "Betting",
    13: "Jewellery & Watches",
    14: "Musical Instruments",
    15: "Office Products",
    16: "Pet Supplies",
    17: "Computer Software",
    18: "Sports & Outdoors",
    19: "Toys & Games",
    20: "Dating Sites",
}


def pick_random_ads(limit: int = 5, seed: int = 42) -> list[dict]:
    """Pick random ad images from different category folders."""
    random.seed(seed)

    # Collect all available (folder_num, image_path) pairs
    all_ads = []
    for root in ADS_ROOTS:
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            try:
                folder_num = int(folder.name)
            except ValueError:
                continue
            for img in folder.glob("*.png"):
                all_ads.append((folder_num, img))
            for img in folder.glob("*.jpg"):
                all_ads.append((folder_num, img))

    # Pick from different categories if possible
    by_category = {}
    for folder_num, img_path in all_ads:
        by_category.setdefault(folder_num, []).append(img_path)

    selected = []
    categories = list(by_category.keys())
    random.shuffle(categories)

    for cat_num in categories[:limit]:
        img_path = random.choice(by_category[cat_num])
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        fmt = "png" if img_path.suffix.lower() == ".png" else "jpeg"
        selected.append({
            "image_id": f"folder{cat_num}_{img_path.stem}",
            "image_base64": b64,
            "image_format": fmt,
            "ground_truth": FOLDER_TO_CATEGORY[cat_num],
        })

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--limit", type=int, default=5, help="Number of ads to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user", default=TEST_USER_ID)
    args = parser.parse_args()

    print("=" * 60)
    print("Testing image ranking agent (classify + score + validate)")
    print("=" * 60)

    # 1. Pick random ads
    print(f"\n[1] Picking {args.limit} random ads from ADS16 folders...")
    ads = pick_random_ads(limit=args.limit, seed=args.seed)
    if not ads:
        print("  ✗ No ads found!")
        sys.exit(1)
    for ad in ads:
        print(f"    {ad['image_id']} -> ground truth: {ad['ground_truth']}")

    # 2. Build user profile
    print(f"\n[2] Building profile for {args.user}...")
    corpus_root = find_corpus_root(args.user)
    profile = build_user_profile(args.user, corpus_root)
    print(f"  ✓ {profile['inf']['gender']}, age {profile['inf']['age']}, {profile['inf']['job']}")

    # 3. Deploy
    print(f"\n[3] Deploying {AGENT_NAME}...")
    handler_code = AGENT_SRC.read_text()
    runtime = deploy_agent(
        name=AGENT_NAME,
        handler_code=handler_code,
        extra_requirements=["strands-agents", "bedrock-agentcore"],
    )
    print(f"  ✓ {runtime['name']} -> {runtime['status']}")
    print(f"    ARN: {runtime['arn']}")

    # 4. Invoke (strip ground_truth from payload)
    print(f"\n[4] Invoking agent with {len(ads)} images...")
    payload_images = [
        {k: v for k, v in ad.items() if k != "ground_truth"}
        for ad in ads
    ]
    result = invoke_agent(runtime["arn"], {
        "user_id": args.user,
        "profile": {
            "inf": profile["inf"],
            "pref": profile["pref"],
            "pos_labels": profile["pos_labels"],
            "neg_labels": profile["neg_labels"],
        },
        "images": payload_images,
    })

    # 5. Validate classifications
    print("\n[5] Classification results (vs ground truth):")
    print(f"    {'Image ID':<20} {'Predicted':<25} {'Ground Truth':<25} {'Match'}")
    print(f"    {'-'*20} {'-'*25} {'-'*25} {'-'*5}")

    correct = 0
    classifications = result.get("classifications", [])
    for i, c in enumerate(classifications):
        gt = ads[i]["ground_truth"]
        predicted = c.get("category", "?")
        match = "✓" if predicted == gt else "✗"
        if predicted == gt:
            correct += 1
        print(f"    {c.get('image_id', '?'):<20} {predicted:<25} {gt:<25} {match}")

    print(f"\n    Accuracy: {correct}/{len(classifications)} ({100*correct/max(len(classifications),1):.0f}%)")

    # 6. Show scores
    print("\n[6] User-ad scores:")
    for s in result.get("scores", []):
        print(f"    {s.get('image_id', '?')}: {s.get('category', '?')} -> {s.get('score', '?')} ({s.get('reasoning', '')})")

    # 7. Cleanup
    if not args.no_cleanup:
        print(f"\n[7] Deleting agent...")
        deleted = delete_agent(AGENT_NAME)
        print(f"  ✓ Deleted: {deleted}")
    else:
        print(f"\n[7] Skipping cleanup. Runtime: {runtime['id']}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
