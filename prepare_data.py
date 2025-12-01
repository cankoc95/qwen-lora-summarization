import pathlib
from datasets import load_dataset

if __name__ == "__main__":
    output_data_dir = "data"
    raw_data = load_dataset("knkarthick/samsum")

    for split in ["train", "validation", "test"]:
        if not pathlib.Path(f"{output_data_dir}/{split}").exists():
            raw_data[split].save_to_disk(pathlib.Path(f"{output_data_dir}/{split}"))

