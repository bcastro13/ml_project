def undersample(df):
    """
    Take random sample of majority class equal to length of minority class
    ~ only accepts binary classes ~

    Parameters
    -----------
    df: DataFrame
        Unsampled DataFrame

    Returns
    --------
    df_sampled: DataFrame
    """

    df_sampled = df[df["class"] == 0].sample(len(df[df["class"] == 1]))
    df_sampled = df_sampled.append(df[df["class"] == 1])
    return df_sampled.sample(frac=1)
