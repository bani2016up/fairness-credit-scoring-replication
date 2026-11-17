"""Train-fitted feature transformations."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    numeric: tuple[str, ...], categorical: tuple[str, ...]
) -> ColumnTransformer:
    numbers = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categories = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    missing_values=None,
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numbers, list(numeric)),
            ("categorical", categories, list(categorical)),
        ],
        sparse_threshold=1.0,
    )


def prepare_fold(preprocessing, features, train, evaluation):
    train_frame = features.iloc[train]
    evaluation_frame = features.iloc[evaluation]
    return preprocessing.fit_transform(train_frame), preprocessing.transform(
        evaluation_frame
    )
