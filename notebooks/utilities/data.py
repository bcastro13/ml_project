import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # surpress warning

PATH = "../../processed_stopwords.csv"


def complete(path=PATH):
    df = pd.read_csv(path)
    return df.dropna()


def neutral(path=PATH):
    df = pd.read_csv(path)
    df = df.dropna()
    df_neutral = df[df["class"] == 2]
    df_other = df[~(df["class"] == 2)]

    df_neutral["class"] = np.zeros(df_neutral.shape[0], dtype="int")
    df_other["class"] = np.ones(df_other.shape[0], dtype="int")

    df_bin = df_neutral.append(df_other)
    df_bin = df_bin[~(df_bin["text"].str.len() >= 2048)]

    fixed_df = df_bin[
        (df_bin["class"] == 0)
        & (
            df_bin["text"].str.contains("bitch")
            | df_bin["text"].str.contains("fuck")
            | df_bin["text"].str.contains("nigger")
            | df_bin["text"].str.contains("nigga")
            | df_bin["text"].str.contains(" ass ")
            | df_bin["text"].str.contains("asshole")
            | df_bin["text"].str.contains("pussy")
            | df_bin["text"].str.contains("chink")
            | df_bin["text"].str.contains("bastard")
            | df_bin["text"].str.contains("dick ")
            | df_bin["text"].str.contains("dicks ")
            | df_bin["text"].str.contains("fag")
            | df_bin["text"].str.contains("damn")
            | df_bin["text"].str.contains("cunt")
            | df_bin["text"].str.contains("shit")
            | df_bin["text"].str.contains("beaner")
        )
    ]
    fixed_df["class"] = fixed_df["class"].replace(0, 1)

    neutral = df_bin[
        (df_bin["class"] == 0)
        & ~(
            df_bin["text"].str.contains("bitch")
            | df_bin["text"].str.contains("fuck")
            | df_bin["text"].str.contains("nigger")
            | df_bin["text"].str.contains("nigga")
            | df_bin["text"].str.contains(" ass ")
            | df_bin["text"].str.contains("asshole")
            | df_bin["text"].str.contains("pussy")
            | df_bin["text"].str.contains("chink")
            | df_bin["text"].str.contains("bastard")
            | df_bin["text"].str.contains("dick ")
            | df_bin["text"].str.contains("dicks ")
            | df_bin["text"].str.contains("fag")
            | df_bin["text"].str.contains("damn")
            | df_bin["text"].str.contains("cunt")
            | df_bin["text"].str.contains("shit")
            | df_bin["text"].str.contains("beaner")
        )
    ]

    not_neutral_df = fixed_df.append(df_bin[df_bin["class"] == 1])
    return not_neutral_df.append(neutral).sample(frac=1)


def hate(path=PATH):
    df = pd.read_csv(path)
    return df[~(df["class"] == 2)]
