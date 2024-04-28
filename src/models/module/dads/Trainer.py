class Trainer(object):
    """ Train the agent
    @property config: configuration of training
    @property agent: SAC agent to be trained
    @property environment: ad defined in anomaly_detection.py
    """

    def __init__(self, config, agent, environment):
        self.config = config
        self.agent = agent
        self.environment = environment

    def run_game_for_agent(self):
        """Run the training process several times to calculate the mean and variance of AUC_PR and AUC_ROC"""
        agent = self.agent(self.config, self.environment)
        correct_search_num, upper_search_num, searched_anomalies = agent.run_n_episodes()  # run a single training process

        if upper_search_num != 0 and searched_anomalies != 0:
            return correct_search_num / searched_anomalies, correct_search_num / upper_search_num
        else:
            return 0, 0
