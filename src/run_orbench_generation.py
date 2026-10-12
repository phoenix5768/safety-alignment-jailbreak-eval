import argparse
import csv
import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )


    completed_ids = set()

    if os.path.exists(args.output):
        existing = pd.read_csv(args.output)

        if "id" in existing.columns:
            completed_ids = set(existing["id"].tolist())

        print(
            f"Resuming: {len(completed_ids)} "
            "examples already completed."
        )

    output_exists = os.path.exists(args.output)

    with open(
        args.output,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "prompt",
                "category",
                "response",
            ],
        )

        if not output_exists:
            writer.writeheader()

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
        ):

            if idx in completed_ids:
                continue

            prompt = row["prompt"]

            for attempt in range(3):
                try:
                    result = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    )

                    response = (
                        result
                        .choices[0]
                        .message
                        .content
                    )

                    break

                except Exception as e:
                    print(
                        f"\nError on example {idx}: {e}"
                    )

                    if attempt == 2:
                        raise

                    time.sleep(5)

            writer.writerow(
                {
                    "id": idx,
                    "prompt": prompt,
                    "category": row["category"],
                    "response": response,
                }
            )

            f.flush()


if __name__ == "__main__":
    main()
