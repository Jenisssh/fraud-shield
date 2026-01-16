"""Tests for the train/val/test split strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_shield.data.splits import stratified_random_split, time_aware_split


class TestStratifiedRandomSplit:
    def test_returns_three_dataframes(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = stratified_random_split(tiny_fraud_df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_total_row_count_preserved(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = stratified_random_split(tiny_fraud_df)
        assert len(train) + len(val) + len(test) == len(tiny_fraud_df)

    def test_approximate_proportions(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = stratified_random_split(tiny_fraud_df, test_size=0.20, val_size=0.15)
        n = len(tiny_fraud_df)
        # train_test_split rounds; allow 1 row of slack
        assert abs(len(test) - 0.20 * n) <= 1
        assert abs(len(val) - 0.15 * n) <= 1
        assert abs(len(train) - 0.65 * n) <= 2

    def test_class_ratio_preserved_across_partitions(self, tiny_fraud_df: pd.DataFrame) -> None:
        overall = tiny_fraud_df["Class"].mean()
        train, val, test = stratified_random_split(tiny_fraud_df)
        for partition in (train, val, test):
            assert abs(partition["Class"].mean() - overall) < 0.01

    def test_no_row_overlap_across_partitions(self, tiny_fraud_df: pd.DataFrame) -> None:
        df = tiny_fraud_df.copy()
        df["__row_id__"] = range(len(df))
        train, val, test = stratified_random_split(df, target="Class")
        ids = {
            "train": set(train["__row_id__"]),
            "val": set(val["__row_id__"]),
            "test": set(test["__row_id__"]),
        }
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        assert len(ids["train"] | ids["val"] | ids["test"]) == len(df)

    def test_reproducible_with_same_seed(self, tiny_fraud_df: pd.DataFrame) -> None:
        a = stratified_random_split(tiny_fraud_df, random_state=7)
        b = stratified_random_split(tiny_fraud_df, random_state=7)
        for left, right in zip(a, b, strict=True):
            pd.testing.assert_frame_equal(left, right)

    def test_different_seeds_produce_different_splits(self, tiny_fraud_df: pd.DataFrame) -> None:
        a_train, _, _ = stratified_random_split(tiny_fraud_df, random_state=1)
        b_train, _, _ = stratified_random_split(tiny_fraud_df, random_state=2)
        assert not a_train.equals(b_train)

    def test_rejects_oversized_split_request(self, tiny_fraud_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="must be < 1"):
            stratified_random_split(tiny_fraud_df, test_size=0.6, val_size=0.5)

    def test_partition_indices_are_reset(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = stratified_random_split(tiny_fraud_df)
        for partition in (train, val, test):
            assert list(partition.index) == list(range(len(partition)))


class TestTimeAwareSplit:
    def test_returns_three_dataframes(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = time_aware_split(tiny_fraud_df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_total_row_count_preserved(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = time_aware_split(tiny_fraud_df)
        assert len(train) + len(val) + len(test) == len(tiny_fraud_df)

    def test_partitions_are_chronological(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = time_aware_split(tiny_fraud_df)
        assert train["Time"].max() <= val["Time"].min()
        assert val["Time"].max() <= test["Time"].min()

    def test_partitions_are_internally_sorted(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = time_aware_split(tiny_fraud_df)
        for partition in (train, val, test):
            assert partition["Time"].is_monotonic_increasing

    def test_approximate_proportions(self, tiny_fraud_df: pd.DataFrame) -> None:
        train, val, test = time_aware_split(tiny_fraud_df, test_size=0.20, val_size=0.15)
        n = len(tiny_fraud_df)
        assert abs(len(test) - 0.20 * n) <= 1
        assert abs(len(val) - 0.15 * n) <= 1
        assert abs(len(train) - 0.65 * n) <= 2

    def test_deterministic(self, tiny_fraud_df: pd.DataFrame) -> None:
        a = time_aware_split(tiny_fraud_df)
        b = time_aware_split(tiny_fraud_df)
        for left, right in zip(a, b, strict=True):
            pd.testing.assert_frame_equal(left, right)

    def test_works_on_unsorted_input(self, tiny_fraud_df: pd.DataFrame) -> None:
        shuffled = tiny_fraud_df.sample(frac=1.0, random_state=999).reset_index(drop=True)
        train, val, test = time_aware_split(shuffled)
        assert train["Time"].max() <= val["Time"].min()
        assert val["Time"].max() <= test["Time"].min()

    def test_no_row_overlap_across_partitions(self, tiny_fraud_df: pd.DataFrame) -> None:
        df = tiny_fraud_df.copy()
        df["__row_id__"] = range(len(df))
        train, val, test = time_aware_split(df)
        ids_train = set(train["__row_id__"])
        ids_val = set(val["__row_id__"])
        ids_test = set(test["__row_id__"])
        assert ids_train.isdisjoint(ids_val)
        assert ids_train.isdisjoint(ids_test)
        assert ids_val.isdisjoint(ids_test)
        assert len(ids_train | ids_val | ids_test) == len(df)

    def test_rejects_oversized_split_request(self, tiny_fraud_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="must be < 1"):
            time_aware_split(tiny_fraud_df, test_size=0.6, val_size=0.5)

    def test_rejects_missing_time_column(self, tiny_fraud_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="time_column"):
            time_aware_split(tiny_fraud_df, time_column="DoesNotExist")
