class StepLRScheduler:
    def __init__(self, initial_lr=0.005, step_size=2, gamma=0.1):
        """
        Step learning rate scheduler
        
        Args:
            initial_lr: Initial learning rate
            step_size: Number of epochs between LR updates
            gamma: Multiplicative factor for LR reduction
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.current_lr = initial_lr
    
    def get_lr(self, epoch):
        """Get learning rate for current epoch"""
        if epoch > 0 and epoch % self.step_size == 0:
            self.current_lr *= self.gamma
        return self.current_lr