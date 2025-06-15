# You'll need to login using huggingface-cli login

import pandas as pd

from router_poc.settings import DATA_DIR


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chatbot_arena_conversations_path = (
        DATA_DIR / "raw" / "chatbot_arena_conversations.parquet"
    )
    if not chatbot_arena_conversations_path.exists():
        print(f"Downloading {chatbot_arena_conversations_path.name}")
        chatbot_arena_conversations = pd.read_parquet(
            "hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet"
        )
        chatbot_arena_conversations.to_parquet(chatbot_arena_conversations_path)
        del chatbot_arena_conversations

    arena_human_preferences_path = DATA_DIR / "raw" / "arena_human_preferences.parquet"
    if not arena_human_preferences_path.exists():
        print(f"Downloading {arena_human_preferences_path.name}")
        arena_human_preferences = pd.read_csv(
            "hf://datasets/lmarena-ai/arena-human-preference-55k/train.csv"
        )
        arena_human_preferences.to_parquet(arena_human_preferences_path)
        del arena_human_preferences

    ppe_mmlu_pro_best_of_k_path = DATA_DIR / "raw" / "ppe_mmlu_pro_best_of_k.parquet"
    if not ppe_mmlu_pro_best_of_k_path.exists():
        print(f"Downloading {ppe_mmlu_pro_best_of_k_path.name}")
        ppe_mmlu_pro_best_of_k = pd.read_parquet(
            "hf://datasets/lmarena-ai/PPE-MMLU-Pro-Best-of-K/data/train-00000-of-00001.parquet"
        )
        ppe_mmlu_pro_best_of_k.to_parquet(ppe_mmlu_pro_best_of_k_path)
        del ppe_mmlu_pro_best_of_k


if __name__ == "__main__":
    main()
