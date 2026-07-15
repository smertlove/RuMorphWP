import pathlib
import pandas as pd
from tqdm import tqdm


def main():

    curdir = pathlib.Path(__file__).parent.resolve()
    datadir = curdir / "data"
    assert datadir.exists()

    target_name = "sigmorphon.txt"
    datadir.touch(target_name)

    for file in (datadir / "sigmorphon").iterdir():
        df = pd.read_csv(file, sep="\t", names=["word", "split", "_"])

        with open(datadir / target_name, "a+", encoding="utf-8") as file:
            # for word in tqdm(df["word"].dropna().tolist()):
            #     file.write(word.lower() + " ")
            #     file.write(word.upper() + " ")
            #     file.write(word.capitalize() + " ")

            for split in tqdm(df["split"].dropna().tolist()):
                file.write(split.replace(" @@", "|##").lower() + "|")


if __name__ == "__main__":
    main()
