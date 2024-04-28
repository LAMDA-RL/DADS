from src.models.module.dads.SAC import SAC
from src.models.module.dads.Trainer import Trainer
from src.models.module.dads.anomaly_detection import ad


class DADS(object):
    def __init__(self, config, dataset_name):
        """Init DADS instance."""
        self.parameter = config

    def train(self, train_df, valid_df, black_len, white_len, contamination, dataset_name, ground_truth):
        AGENT = SAC
        self.environment = ad(train_df, valid_df, black_len, white_len, contamination, dataset_name, ground_truth,
                              self.parameter["Environment"])
        trainer = Trainer(self.parameter["Agent"], AGENT, self.environment)
        search_acc, search_hit = trainer.run_game_for_agent()

        return search_acc, search_hit

    def evaluate(self, test_df, known_test_df=None, unknown_test_df=None):
        auc_roc, auc_pr, p95, best_episode = self.environment.evaluate(test_df, True)
        if known_test_df is not None:
            auc_roc_known, auc_pr_known, p95_known, _ = self.environment.evaluate(known_test_df, True)
            auc_roc_unknown, auc_pr_unknown, p95_unknown, _ = self.environment.evaluate(unknown_test_df, True)

            return auc_roc, auc_pr, p95, auc_roc_known - auc_roc_unknown, auc_pr_known - auc_pr_unknown, p95_known - p95_unknown, best_episode
        else:
            return auc_roc, auc_pr, p95, best_episode

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device):
        """Load SSAD model from import_path."""
        pass
