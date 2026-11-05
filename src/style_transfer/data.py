import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from datasets import load_dataset
from tqdm.auto import tqdm

COCO_METADATA_URL = (
    "https://huggingface.co/datasets/phiyodr/coco2017/resolve/main/data/"
    "validation-00000-of-00001-e3c37e369512a3aa.parquet"
)
DALI_DATASET = "Artificio/WikiArt"
DALI_PAINTINGS = {
    54723: ("Постоянство памяти", 1931, "the-persistence-of-memory-1931"),
    5123: ("Метаморфозы Нарцисса", 1937, "the-metamorphosis-of-narcissus"),
    14043: ("Лебеди, отражённые в слонах", 1937, "cygnes-refletant-des-elephants"),
    70887: ("Слоны", 1948, "the-elephants-large"),
    21455: ("Искушение святого Антония", 1946, "the-temptation-of-st-anthony"),
    79800: (
        "Сон, вызванный полётом пчелы вокруг граната, за секунду до пробуждения",
        1944,
        "dream-caused-by-the-flight-of-a-bee-around-a-pomegranate-one-second-before-awakening",
    ),
    101508: ("Пылающий жираф", 1937, "flaming-giraffe"),
    53166: (
        "Распад постоянства памяти",
        "1952–1954",
        "the-disintegration-of-the-persistence-of-memory",
    ),
}


def _download_image(url: str, path: Path) -> None:
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())


def download_coco_images(directory: Path, count: int = 24, seed: int = 42) -> list[dict]:
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = metadata["images"]
        if metadata["seed"] == seed and len(records) == count:
            if all((directory / record["file_name"]).exists() for record in records):
                return records

    directory.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "parquet",
        data_files={"validation": COCO_METADATA_URL},
        split="validation",
        cache_dir=str(directory.parent / ".cache"),
    )
    sample = dataset.shuffle(seed=seed).select(range(count))
    records = []
    for item in tqdm(sample, desc="COCO"):
        file_name = Path(item["file_name"]).name
        _download_image(item["coco_url"], directory / file_name)
        records.append(
            {
                "file_name": file_name,
                "image_id": item["image_id"],
                "coco_url": item["coco_url"],
                "width": item["width"],
                "height": item["height"],
                "captions": [caption.strip() for caption in item["captions"]],
            }
        )

    metadata = {"dataset": "COCO 2017", "split": "validation", "seed": seed, "images": records}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def download_dali_images(directory: Path) -> list[dict]:
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = metadata["images"]
        if (
            metadata["dataset"] == DALI_DATASET
            and [record["row_index"] for record in records] == list(DALI_PAINTINGS)
            and all((directory / record["file_name"]).exists() for record in records)
        ):
            return records

    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (title_ru, year, slug) in tqdm(DALI_PAINTINGS.items(), desc="Dali"):
        params = {
            "dataset": DALI_DATASET,
            "config": "default",
            "split": "train",
            "offset": index,
            "length": 1,
        }
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(params)
        with urlopen(url, timeout=60) as response:
            data = json.load(response)
        item = data["rows"][0]["row"]
        file_name = f"wikiart_{index:05d}.jpg"
        _download_image(item["image"]["src"], directory / file_name)
        records.append(
            {
                "file_name": file_name,
                "row_index": index,
                "artist": item["artist"],
                "title": item["title"],
                "title_ru": title_ru,
                "year": year,
                "style": item["style"],
                "source_url": f"https://www.wikiart.org/en/salvador-dali/{slug}",
                "dataset_url": (
                    f"https://huggingface.co/datasets/{DALI_DATASET}"
                    f"/viewer/default/train?row={index}"
                ),
                "width": item["image"]["width"],
                "height": item["image"]["height"],
            }
        )

    metadata = {"dataset": DALI_DATASET, "split": "train", "images": records}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
