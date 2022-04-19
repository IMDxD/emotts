from phonemizer.phonemize import phonemize
import re, regex


## from Tacotron (from SIWIS corpus)
punkt_class = "“”„«»!,.:;–?\"'…\[\](){}—\-"  ## NOTE: without the whitespace
re_punkt_class = re.compile("([" + punkt_class + "])")


def espeak_phon(
    text,
    lang="ru",
    accent=None,
    clean=True,
    add_punkt=True,
    mark_pausation=True,
    pausation_symbol="%",
    _spec_seq="@@@",
    verbose=False,
):
    language = lang
    if accent:
        language += f"-{accent}"
    if verbose:
        print(f"Original input:\n{text}\n")

    punkt = None
    if add_punkt:
        spec_phonseq = phonemize(_spec_seq, language=language, backend="espeak").strip()
        assert len(spec_phonseq) > 0
        assert not re.match("\s", spec_phonseq)
        punkt = re_punkt_class.finditer(text)
        # punkt = re_punkt_class.findall(text)
        # punkt = regex.findall('([' + punkt_class + '])', text)
        punkt = [x.group(0) for x in punkt]
        if verbose:
            print(f"Punctuation recognized:\n{punkt}\n")
        if len(punkt) > 0:
            punkt = iter(punkt)
        else:
            punkt = None
        text = re.sub(re_punkt_class, _spec_seq + r"\1", text)
        if verbose:
            print(f"Input with marked punctuation:\n{text}\n")

    phoneseq = phonemize(text, language=language, backend="espeak")
    if mark_pausation:
        phoneseq = re.sub("\n", pausation_symbol, phoneseq)
    else:
        phoneseq = re.sub("\n", " ", phoneseq)
    if verbose:
        print(f"Input with marked pausation:\n{phoneseq}\n")

    if clean:
        phoneseq = regex.sub(
            "\(.+?\)", "", phoneseq
        )  ## use regex for non-greedy matching
        ## zz = liaison after z
        phoneseq = re.sub("z+", "z", phoneseq)
    if verbose:
        print(f'Input after "cleaning":\n{phoneseq}\n')

    out_phoneseq = ""
    if add_punkt and (punkt is not None):
        split_phoneseq = [x for x in re.split("(" + spec_phonseq + ")", phoneseq) if x]
        for x in split_phoneseq:
            if x == spec_phonseq:
                out_phoneseq += next(punkt)
            else:
                out_phoneseq += x
    else:
        out_phoneseq = phoneseq
    out_phoneseq = re.sub(r" %", r"%", out_phoneseq)
    out_phoneseq = re.sub(r"([" + punkt_class + "]+)(%)", r"\2\1 ", out_phoneseq)
    out_phoneseq = re.sub(r" %", r"%", out_phoneseq)
    out_phoneseq = re.sub("\s+", " ", out_phoneseq)
    if verbose:
        print(f"Output:\n{out_phoneseq}")

    return out_phoneseq
