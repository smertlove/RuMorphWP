import pathlib
import requests

NAME2URL = {
    "train.csv": "https://github.com/sigmorphon/2022SegmentationST/blob/main/data/rus.word.train.tsv",
    "dev.csv": "https://github.com/sigmorphon/2022SegmentationST/blob/main/data/rus.word.dev.tsv",
    "test_gold.csv": "https://github.com/sigmorphon/2022SegmentationST/blob/main/data/rus.word.test.gold.tsv",
    "test.csv": "https://github.com/sigmorphon/2022SegmentationST/blob/main/data/rus.word.test.tsv",
}


def download_file(url: str, dest_path: pathlib.Path, chunk_size: int = 8192) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)


def main():

    curdir = pathlib.Path(__file__).parent.resolve()
    datadir = curdir / "data"
    datadir.mkdir(parents=True, exist_ok=True)

    for filename, url in NAME2URL.items():
        dest_path = datadir / filename
        print(f"Downloading {filename}...", end=" ")

        try:
            download_file(url, dest_path)
            print(f"✓ Downloaded {filename}")
        except requests.RequestException as e:
            print(f"✗ Failed to download {filename}: {e}")


if __name__ == "__main__":
    main()
