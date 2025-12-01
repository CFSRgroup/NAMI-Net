class WarmupAndExponentialDecayScheduler:
    def __init__(self, optimizer, warmup_epochs, max_lr, base_lr, decay_factor, decay_step, num_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        self.num_epochs = num_epochs

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (epoch / (self.warmup_epochs - 1))
        else:
            decay_epoch = (epoch - self.warmup_epochs) // self.decay_step
            lr = self.max_lr * (self.decay_factor ** (decay_epoch + 1))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr