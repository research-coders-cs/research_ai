import torch

RUN_EXAMPLES = True

#---- ^^
# Some convenience helper functions used throughout the notebook

# def is_interactive_notebook():
#     return __name__ == "__main__"

def show_example(fn, args=[]):
    #print("@@ __name__:", __name__)  # transduction

    # if __name__ == "__main__" and RUN_EXAMPLES:
    #     return fn(*args)
    #====
    if RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    # if __name__ == "__main__" and RUN_EXAMPLES:
    #     fn(*args)
    #====
    if RUN_EXAMPLES:
        fn(*args)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None
#---- $$