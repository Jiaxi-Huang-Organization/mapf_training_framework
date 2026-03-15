def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()
    obs, _ = env.reset(seed=env.grid_config.seed)
    step_count = 0
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        # 添加监控代码
        step_count += 1
        dead_count = sum(1 for info in infos if not info.get('is_active', True))
        print(f"Step {step_count}: Dead agents = {dead_count}/{len(infos)}")
        if all(dones) or all(tr):
            break
    return results_holder.get_final()


class ResultsHolder:
    """
    Holds and manages the results obtained during an episode.

    """

    def __init__(self):
        """
        Initializes an instance of ResultsHolder.
        """
        self.results = dict()

    def after_step(self, infos):
        """
        Updates the results with the metrics from the given information.

        Args:
            infos (List[dict]): List of dictionaries containing information about the episode.

        """
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        """
        Returns the final results obtained during the episode.

        Returns:
            dict: The final results.

        """
        return self.results

    def __repr__(self):
        """
        Returns a string representation of the ResultsHolder.

        Returns:
            str: The string representation of the ResultsHolder.

        """
        return str(self.get_final())
