class DQN:
    def c(
        learning_rate: float = 0.0001,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
    ):
        pass


class TD3:
    def c(
        learning_rate: float = 0.001,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
    ):
        pass


class PPO:
    def c(
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
    ):
        pass


class A2C:
    def c(
        learning_rate: float = 0.0007,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 0.00001,
        use_rms_prop: bool = True,
        normalize_advantage: bool = False,
    ):
        pass
