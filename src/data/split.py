"""Create the stratified train/val/test split."""

from sklearn.model_selection import train_test_split

def make_split(df, seed = 42):
    train, rest = train_test_split(
        df,
        test_size = 0.4,
        stratify = df["disaster"],
        random_state = seed
    )

    validation, test = train_test_split(
        rest,
        test_size = 0.5,
        stratify = rest["disaster"],
        random_state = seed
    )

    return train, validation, test