import unittest

from mania_difficulty.train import cross_validation_splits, split_indices_by_group


def groups_for(indices, groups):
    return {groups[index] for index in indices}


class GroupSplitTests(unittest.TestCase):
    def test_group_split_keeps_group_ids_in_one_split(self):
        groups = ["set_a", "set_a", "set_b", "set_b", "set_c", "set_d", "set_e", "set_f"]

        train_indices, val_indices, test_indices = split_indices_by_group(groups, seed=7)

        train_groups = groups_for(train_indices, groups)
        val_groups = groups_for(val_indices, groups)
        test_groups = groups_for(test_indices, groups)
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        self.assertEqual(len(train_indices) + len(val_indices) + len(test_indices), len(groups))

    def test_cross_validation_splits_keep_group_ids_out_of_fold(self):
        groups = ["set_a", "set_a", "set_b", "set_b", "set_c", "set_d", "set_e", "set_f"]

        splits = list(cross_validation_splits(len(groups), groups=groups, folds=3, seed=11))

        self.assertEqual(len(splits), 3)
        for train_indices, val_indices in splits:
            self.assertTrue(groups_for(train_indices, groups).isdisjoint(groups_for(val_indices, groups)))


if __name__ == "__main__":
    unittest.main()
