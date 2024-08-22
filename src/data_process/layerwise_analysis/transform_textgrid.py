import hydra
import omegaconf
from pathlib import Path
from tqdm import tqdm
import textgrids


def write_to_file(write_str: str, fname: Path):
    with open(str(fname), "w") as f:
        f.write(write_str)


class Preprocessor:
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        self.cfg = cfg

    def save_data(self):
        token_lst_dct = self.read_data()
        self.get_token_alignment_ordered_lst(token_lst_dct)

    def get_token_alignment_ordered_lst(
        self, token_lst_dct: dict[str, list[str]], data_dir, dataset_split
    ):
        """
        Save the list of token-level alignments to a tsv file
        """
        for key, value in token_lst_dct.items():
            write_str = []
            for item in tqdm(value):
                write_str.append("\t".join(item.split(" ")))
            write_fn = (
                Path(self.cfg.path.kablab.forced_alignment_results_dir).expanduser()
                / "token_level_alignments"
            )
            write_to_file("\n".join(write_str), write_fn)

    def read_data(self) -> dict[str, list[str]]:
        wrd_lst: list[str] = []
        phn_lst: list[str] = []
        data_dir = (
            Path(self.cfg.path.kablab.forced_alignment_results_dir).expanduser()
            / "results"
        )
        data_path_list = list(data_dir.glob("**/*.TextGrid"))
        for data_path in tqdm(data_path_list):
            self.get_info(data_path, phn_lst, wrd_lst)
        token_lst_dct = {"phone": phn_lst, "word": wrd_lst}
        return token_lst_dct

    def txt_from_tier(
        self, tier_content: textgrids.Tier, data_lst: list[str], fname: str, unit: str
    ) -> None:
        """
        Save as filename start end label
        """
        for item in tier_content:
            label = item.text
            if label:  # check that it is non-empty
                start = str(item.xmin)
                end = str(item.xmax)
                text_out = " ".join([fname, start, end, label])
                if label not in ["spn", "sp"]:
                    data_lst.append(text_out)

    def get_info(self, data_path: Path, phn_lst: list[str], wrd_lst: list[str]) -> None:
        grid = textgrids.TextGrid(str(data_path))
        fname = data_path.stem
        self.txt_from_tier(grid["phones"], phn_lst, fname, "phone")
        self.txt_from_tier(grid["words"], wrd_lst, fname, "word")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    processor = Preprocessor(cfg)
    processor.save_data()


if __name__ == "__main__":
    main()
