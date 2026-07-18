from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast
import pathlib

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
CONTINUING_SUBWORD_PREFIX = "##"
POSES_FEATS = ['ADJ',
 'ADP',
 'ADV',
 'ADVPRO',
 'ANUM',
 'AUX',
 'CCONJ',
 'COM',
 'DET',
 'INIT',
 'INTJ',
 'NOUN',
 'NUM',
 'PARENTH',
 'PART',
 'PRED',
 'PREDPRO',
 'PRON',
 'PROPN',
 'PUNCT',
 'SCONJ',
 'SYM',
 'VERB',
 'X',
 'Abbr:Yes',
 'Animacy:Anim',
 'Animacy:Inan',
 'Anom:Yes',
 'Aspect:Imp',
 'Aspect:Perf',
 'Case:Acc',
 'Case:Acc2',
 'Case:Dat',
 'Case:Gen',
 'Case:Ins',
 'Case:Loc',
 'Case:Loc2',
 'Case:Nom',
 'Case:Nom1',
 'Case:Par',
 'Case:Voc',
 'Clitic:Yes',
 'Degree:Cmp',
 'Degree:Cmp2',
 'Degree:Pos',
 'Degree:Sup',
 'Foreign:Yes',
 'Gender:Fem',
 'Gender:Masc',
 'Gender:Neut',
 'Hyph:Yes',
 'InflClass:Ind',
 'Mood:Cnd',
 'Mood:Imp',
 'Mood:Imp2',
 'Mood:Ind',
 'NameType:Com',
 'NameType:Evn',
 'NameType:Geo',
 'NameType:Giv',
 'NameType:Oth',
 'NameType:Pat',
 'NameType:Pro',
 'NameType:Prs',
 'NameType:Sur',
 'NameType:Zoo',
 'NumForm:Combi',
 'NumForm:Digit',
 'NumForm:Roman',
 'NumForm:Word',
 'NumType:Card',
 'NumType:Frac',
 'NumType:Ord',
 'NumType:Sets',
 'Number:Count',
 'Number:Dual',
 'Number:Plur',
 'Number:Sing',
 'Person:1',
 'Person:2',
 'Person:3',
 'Polarity:Neg',
 'Poss:Yes',
 'PronType:Dem',
 'PronType:Emp',
 'PronType:Exc',
 'PronType:Ind',
 'PronType:Int',
 'PronType:Neg',
 'PronType:Prs',
 'PronType:Rcp',
 'PronType:Rel',
 'PronType:Tot',
 'Reflex:Yes',
 'Tense:Aor',
 'Tense:Fut',
 'Tense:Imp',
 'Tense:Past',
 'Tense:Pres',
 'Transit:Intr',
 'Transit:Intr,Tran',
 'Transit:Tran',
 'Typo:Yes',
 'Variant:Short',
 'VerbForm:Conv',
 'VerbForm:Fin',
 'VerbForm:Inf',
 'VerbForm:Part',
 'Voice:Act',
 'Voice:Act,Pass',
 'Voice:Mid',
 'Voice:Pass']

def build_wordpiece_tokenizer(lowercase: bool) -> Tokenizer:
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="<unk>",
            continuing_subword_prefix=CONTINUING_SUBWORD_PREFIX,
        )
    )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFC(),
            normalizers.Replace(pattern=r"\s+", content=" "),
            normalizers.BertNormalizer(
                clean_text=True,
                handle_chinese_chars=False,
                strip_accents=False,
                lowercase=True,
            ),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern="|", behavior="removed"),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Whitespace(),
        ]
    )

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[
            ("<s>", SPECIAL_TOKENS.index("<s>")),
            ("</s>", SPECIAL_TOKENS.index("</s>")),
        ],
    )

    tokenizer.decoder = decoders.WordPiece(prefix=CONTINUING_SUBWORD_PREFIX)
    return tokenizer


def train_bart_tokenizer(
    file_path: str,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    lowercase: bool = False,
    save_path: str = None,
) -> Tokenizer:

    tokenizer = build_wordpiece_tokenizer(lowercase=lowercase)

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS + POSES_FEATS,
        show_progress=True,
        continuing_subword_prefix=CONTINUING_SUBWORD_PREFIX,
    )

    tokenizer.train(
        [
            file_path,
        ],
        trainer,
    )

    current_vocab = tokenizer.get_vocab()

    other_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    )

    tokens_to_add = []
    for char in other_chars:
        if char not in current_vocab:
            tokens_to_add.append(char)
    # print(len(tokens_to_add))
    # tokens_to_add = list(set(tokens_to_add))
    # print(len(tokens_to_add))
    tokenizer.add_tokens(tokens_to_add)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens = POSES_FEATS,
        model_max_length=1024,
        name_or_path=save_path,
    )

    hf_tokenizer.save_pretrained(save_path)

    return hf_tokenizer


if __name__ == "__main__":
    datapath = pathlib.Path("./data/segments_split_roots_flat.txt")
    assert datapath.exists()

    tokenizers_path = pathlib.Path("./tokenizers")
    tokenizers_path.mkdir(exist_ok=True)

    train_bart_tokenizer(
        file_path=str(datapath), save_path=str(tokenizers_path / "hybrid-wp")
    )
