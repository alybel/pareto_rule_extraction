import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RuleExtractor:
    def __init__(self, estimator, feature_names=None, class_names=None, show_progress=False, debug=0):
        self.show_progress = show_progress
        if type(estimator) == Pipeline:
            estimator = estimator.steps[-1][-1]  # last step is the tuple of estimator ('clf', some_estimator)

        self.estimator = estimator
        self.feature_names = feature_names
        self.n_features = self.estimator.n_features_
        self.class_names = class_names
        self.debug = debug
        self.predictions = []
        self.samples = []


        if type(estimator) == RandomForestClassifier:
            self.random_forest_type = "classifier"
            names = ["RULE_ID", "RULE_NAME", "DIRECTION_NAME", "TREE_ID", "NODE_ID", "FEATURE_ID"]
            if self.feature_names is not None:
                names.append("FEATURE_NAME")
            names.extend(["DIRECTION", "THRESHOLD", "N_SAMPLES"])
            if self.class_names is None:
                self.class_names = self.estimator.classes_.tolist()
            names.extend(["P_%s" % self.class_names[i] for i in range(len(self.class_names))])
        elif type(estimator) == RandomForestRegressor:
            self.random_forest_type = "regressor"
            names = ["RULE_ID", "RULE_NAME", "DIRECTION_NAME", "TREE_ID", "NODE_ID", "FEATURE_ID"]
            if self.feature_names is not None:
                names.append("FEATURE_NAME")
            names.extend(["DIRECTION", "THRESHOLD", "N_SAMPLES", "VALUE"])
        else:
            raise TypeError("The passed Classifier or Regressor is a %s and not a RandomForest. "
                            "You need to pass a sklearn.ensemble.RandomForestClassifier or "
                            "sklearn.ensemble.RandomForestClassifier to use this function." % (str(type(estimator))))
        self.names = names
        self.extracted_rules = None
        self.rule_statistics = None
        if self.show_progress:
            print("Finished intialisation.")

    def _get_leaves(self, tree):
        """
        This function is called within get_rules
        :param tree:
        :return:
        """
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        leave_ids = [idx for idx, is_leave in enumerate(is_leaves) if is_leave]
        return leave_ids

    def _get_rules(self, tree, idx):
        """
        this function is called for each tree
        :param tree:
        :param idx:
        :return:
        """

        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        samples = tree.n_node_samples
        values = tree.value
        leave_ids = self._get_leaves(tree)

        all_rules = []
        if self.show_progress:
            print("Tree %d has %d leaves." % (idx, len(leave_ids)))
        leave_counter = 0
        for leave_id in leave_ids:
            if self.show_progress:
                print("Extracted rules for %d leaves." % leave_counter, end="\r")
                leave_counter += 1
            current_node = leave_id
            if self.feature_names is not None:
                rules = [[idx, current_node, None, None, None, None, samples[current_node]]]
            else:
                rules = [[idx, current_node, None, None, None, samples[current_node]]]
            rules[-1].extend(values[current_node][0])
            for i in reversed(range(leave_id)):
                if children_left[i] == current_node:
                    current_node = i
                    rule = [idx, current_node, feature[current_node]]
                    if self.feature_names is not None:
                        rule.append(self.feature_names[feature[current_node]])
                    rule.extend(["leq", threshold[current_node], samples[current_node]])
                    rule.extend(values[current_node][0])
                    rules.append(rule)
                elif children_right[i] == current_node:
                    current_node = i
                    rule = [idx, current_node, feature[current_node]]
                    if self.feature_names is not None:
                        rule.append(self.feature_names[feature[current_node]])
                    rule.extend(["g", threshold[current_node], samples[current_node]])
                    rule.extend(values[current_node][0])
                    rules.append(rule)
            all_rules.append(list(reversed(rules)))
        return all_rules

    def extract_rules(self):
        """
        Loop over tree estimators and extract all rules from all trees in the forest. All rules with sample sizes at nodes
        and threshold values are returned in a dataframe. No aggregation happens on this level. The object variable
        self.extracted_rules is set here.
        This is the first necessary step to apply the rule aggregation mechanism.
        :return: extracted_rules
        """
        if self.debug:
            print('running extract_rules')
        estimators = self.estimator.estimators_
        all_rules = []
        for idx, tree in enumerate(estimators):
            if self.show_progress:
                print("Started extracting rules for tree %d." % idx)
            tree = tree.tree_
            rules = self._get_rules(tree, idx)
            all_rules.extend(rules)
        df = []
        for idx, rule in enumerate(all_rules):
            rule_id = idx
            if self.feature_names is not None:
                rule_name = "-".join([str(step[3]) for step in rule[:-1] if step[3] is not None])
                dir_name = "-".join([str(step[4]) for step in rule[:-1] if step[4] is not None])
            else:
                rule_name = "-".join([str(step[2]) for step in rule[:-1] if step[2] is not None])
                dir_name = "-".join([str(step[3]) for step in rule[:-1] if step[3] is not None])
            for step in rule:
                if self.random_forest_type == "classifier":
                    if self.feature_names is not None:
                        step, p = step[0:7], step[7:]
                    else:
                        step, p = step[0:6], step[6:]
                    sum_p = sum(p)
                    p = [i / sum_p for i in p]
                    step.extend(p)
                rule_step = [rule_id, rule_name, dir_name]
                rule_step.extend(step)
                df.append(rule_step)

        df = pd.DataFrame(df, columns=self.names)
        self.extracted_rules = df
        return df

    def get_feature_counts(self):
        if self.debug:
            print('running get_feature_counts')
        if self.feature_names:
            return self.extracted_rules['FEATURE_NAME'].value_counts()
        else:
            return self.extracted_rules['FEATURE_ID'].value_counts()

    def get_rule_counts(self, top_n=None):
        """
        Count per Rule (RULE_NAME) and Direction how often it was found. Return a list of distinct rules.
        Result is ordered by rule occurrence in a descending fashion.

        :return:
        """
        # get all possible RULE_/DIRECTION_NAMEs and sort them by count of trees they are used in
        # since the features are stored in different rows for one rule, each RULE_ID has RULE_NAME and DIRECTION_NAME
        # to find the ones with the same Features and Directions
        if self.debug:
            print('running get_rule_counts')
        if self.extracted_rules is None:
            self.extract_rules()
        distinct_rules = self.extracted_rules[["RULE_ID", "RULE_NAME", "DIRECTION_NAME", "TREE_ID"]].drop_duplicates()
        distinct_rules = distinct_rules.groupby(['RULE_NAME', "DIRECTION_NAME"])["TREE_ID"].count().reset_index(
            name="N_TREES_WITH_RULE")
        distinct_rules = distinct_rules.sort_values(by="N_TREES_WITH_RULE")
        distinct_rules = distinct_rules[::-1]  # highest first
        if top_n is None:
            return distinct_rules
        elif top_n < 1:
            n_cut = int(len(distinct_rules) * top_n)
            return distinct_rules.iloc[:n_cut]
        elif top_n >= 1:
            return distinct_rules.iloc[:top_n]

    def extract_rule_statistics(self, weighted=True, use_median=False, top_n=None):
        """
        A new dataframe is produced on top of
        :param weighted:
        :param use_median:
        :return:
        """
        if self.debug:
            print('running extract_rule_statistics')

        # new dataframe for statistics with the following columns
        if self.feature_names is not None:
            columns = ["RULE_DIRECTION_ID", "RULE_NAME", "DIRECTION_NAME", "COUNT", "FEATURE_ID", "FEATURE_NAME",
                       "DIRECTION", "THRESHOLD",
                       "THRESHOLD_VAR", "SAMPLES", "SAMPLES_VAR", "VALUE", "VALUE_VAR"]
        else:
            columns = ["RULE_DIRECTION_ID", "RULE_NAME", "DIRECTION_NAME", "COUNT", "FEATURE_ID", "DIRECTION",
                       "THRESHOLD",
                       "THRESHOLD_VAR", "SAMPLES", "SAMPLES_VAR", "VALUE", "VALUE_VAR"]
        ruledf = pd.DataFrame(columns=columns)

        # get the distinct rules and their count
        distinct_rules = self.get_rule_counts(top_n=top_n)

        # extract statistics for all distinct rules
        new_rule_id = 0
        for row in distinct_rules.itertuples():
            # get all the rules for the current RULE_/DIRECTION_NAME
            rule_name = getattr(row, "RULE_NAME")
            dir_name = getattr(row, "DIRECTION_NAME")

            # this is probably not the fastest way, but features and directions are stored in different rows,
            # so comparing the RULE_/DIRECTION_NAMES is easier
            rules = self.extracted_rules.loc[
                (self.extracted_rules["RULE_NAME"] == rule_name) & (self.extracted_rules["DIRECTION_NAME"] == dir_name)]

            # get the ids of those rules
            ids = list(set(rules["RULE_ID"]))

            # all these ids have the same rule, so to get features and directions, we get them from the first rule
            first_rule = self.extracted_rules.loc[self.extracted_rules["RULE_ID"] == ids[0]]
            directions = [r for r in list(first_rule["DIRECTION"]) if r is not None]
            if self.feature_names is not None:
                features = [r for r in list(first_rule["FEATURE_NAME"]) if r is not None]
            feature_ids = [r for r in list(first_rule["FEATURE_ID"]) if r is not None]

            # init dict for this rule
            samples = []
            thresholds = []
            values = []
            count = 0


            # iterate through ids and collect information
            for curr_id in ids:
                # get full rules for this id from the actual df
                rules = self.extracted_rules.loc[self.extracted_rules["RULE_ID"] == curr_id]

                # get the needed information
                samples.append([r for r in list(rules["N_SAMPLES"]) if r is not None])
                thresholds.append([r for r in list(rules["THRESHOLD"]) if r is not None])
                if self.random_forest_type == 'regressor':
                    values.append([r for r in list(rules["VALUE"]) if r is not None])
                else:
                    values.append([r for r in list(rules["P_True"]) if r is not None])
                count += 1

            # iterate through steps/features of the rule to bring information in the correct format
            for j in range(len(directions)):
                # get the needed information
                step_thresholds = [item[j] for item in thresholds]
                step_samples = [item[j] for item in samples]

                # compute average and variance of n_samples
                av_samp = np.average(step_samples)
                var_samp = np.var(step_samples)

                # compute average and variance for thresholds (weighted by samples if weighted=True)
                if weighted:
                    average = np.average(step_thresholds, weights=step_samples)
                    variance = np.average((step_thresholds - average) ** 2, weights=step_samples)
                else:
                    average = np.average(step_thresholds)
                    variance = np.var(step_thresholds)

                # append this step of the rule to the dataframe
                if self.feature_names is not None:
                    ruledf = ruledf.append(
                        pd.DataFrame(
                            [[new_rule_id, rule_name, dir_name, count, feature_ids[j], features[j], directions[j],
                              average, variance, av_samp, var_samp, None, None]], columns=columns))
                else:
                    ruledf = ruledf.append(
                        pd.DataFrame([[new_rule_id, rule_name, dir_name, count, features[j], directions[j],
                                       average, variance, av_samp, var_samp, None, None]], columns=columns))

            # last step is the leaf
            leaf_values = [item[-1] for item in values]
            leaf_samples = [item[-1] for item in samples]
            leaf_av_samp = np.average(leaf_samples)
            leaf_var_samp = np.var(leaf_samples)
            if weighted:
                leaf_average = np.average(leaf_values, weights=leaf_samples)
                leaf_variance = np.average((leaf_values - leaf_average) ** 2, weights=leaf_samples)
            else:
                leaf_average = np.average(leaf_values)
                leaf_variance = np.var(leaf_values)
            if self.feature_names is not None:
                ruledf = ruledf.append(
                    pd.DataFrame([[new_rule_id, rule_name, dir_name, count, None, None, None, None, None,
                                   leaf_av_samp, leaf_var_samp, leaf_average, leaf_variance]], columns=columns))
            else:
                ruledf = ruledf.append(pd.DataFrame([[new_rule_id, rule_name, dir_name, count, None, None, None, None,
                                                      leaf_av_samp, leaf_var_samp, leaf_average, leaf_variance]],
                                                    columns=columns))
            new_rule_id += 1
        self.rule_statistics = ruledf
        return ruledf


      
    def apply_signum_to_predictions(self, default_prediction = 0):
      preds =[]
      for pred in self.predictions:
        pred = [np.sign(e) if e is not None else default_prediction for e in pred]
        preds.append(pred)
      self.predictions = preds
      
    def predict_for_top_n(self, n, weighted=True, default_prediction=0):
        y_pred = []
        # for each sample
        for idx, pred in enumerate(self.predictions):
            # get the predictions of this sample for the top_n rules to compute average (dismiss Nones)
            pred = [i for i in pred[:n] if i is not None]

            if len(pred) == 0:
                y_pred.append(default_prediction)

            else:
                if weighted:
                    # samp are the number of samples for each rule which had this rule in training (used as weights)
                    samp = [i for i in self.samples[idx][:n] if i is not None]
                    av = np.average(pred, weights=samp)
                else:
                  print(pred)
                  av = np.average(pred)
                y_pred.append(av)
        return y_pred

    def predict_samples(self, samples, with_var=False):
        self.predictions = []
        self.samples = []
        for sample in samples:
            ret = self.predict_sample(sample, with_var=with_var)
            self.predictions.append(ret[0])
            self.samples.append(ret[1])
        return self

    def predict_sample(self, sample, with_var=False, default_prediction=0):
        # init
        predictions = []
        samples = []
        curr_id = 0
        rule_fits = True

        # iterate through the rows of rule_statistics, one rule is more than one row
        for row in self.rule_statistics.itertuples():
            # check if were still in the same rule, if not, reset rule_fits
            prev_id = curr_id
            curr_id = getattr(row, "RULE_DIRECTION_ID")
            if curr_id != prev_id:
                # new rule
                rule_fits = True

            # check if the rule fits (which means if the current sample can follow the directions for the features of this rule)
            if rule_fits:
                # check for the current feature if it is None, which means it is a leaf, so the rule actually fitted
                curr_feature = getattr(row, "FEATURE_ID")
                if curr_feature is not None:
                    # get threshold and direction
                    curr_threshold = getattr(row, "THRESHOLD")
                    curr_direction = getattr(row, "DIRECTION")
                    # get the feature value and check if the rule fits
                    curr_feature = int(curr_feature)
                    this_value = sample[curr_feature]
                    if curr_feature is not None:
                        # does the rule fit? check for smaller equal or greater depending on direction
                        if curr_direction == "leq":
                            # if we allow variance, compare if smaller or equal with the threshold + variance
                            if with_var:
                                curr_var = getattr(row, "THRESHOLD_VAR")
                                curr_threshold += curr_var
                            if not this_value <= curr_threshold:
                                rule_fits = False
                        else:
                            # if we allow variance, compare if greater with the threshold - variance
                            if with_var:
                                curr_var = getattr(row, "THRESHOLD_VAR")
                                curr_threshold -= curr_var
                            if not this_value >= curr_threshold:
                                rule_fits = False
                                predictions.append(None)
                                samples.append(None)
                else:
                    # reached end of rule and it still fits, so we append prediction value to output
                    curr_value = getattr(row, "VALUE")
                    predictions.append(curr_value)

                    # also append samples count, to compute weighted if needed
                    curr_samples = getattr(row, "SAMPLES")
                    samples.append(curr_samples)
        return (predictions, samples)
