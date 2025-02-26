try:
    from IPython import get_ipython
    if get_ipython():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm


class ProgressBars:
    def __init__(self, desc="Processing", total=None, mininterval=0.5, ncols=100, leave=False, position=0):
        self.desc = desc
        self.total = total
        self.mininterval = mininterval
        self.ncols = ncols
        self.leave = leave
        self.position = position
        self.active_progress = {}

    def add_task(self, task, total, completed=0, acc = None):
        self.active_progress[task] = tqdm(
            desc=task,
            total=total,
            mininterval=self.mininterval,
            ncols=self.ncols,
            leave=self.leave,
            position=self.position,
            dynamic_ncols=True
        )
        self.active_progress[task].n = completed
        if acc is not None:
            self.active_progress[task].set_postfix_str(f"Best Accuracy: {acc:0.3f}")
        self.active_progress[task].refresh()
        return task
    
    def remove_task(self, task):
        self.active_progress[task].close()
        del self.active_progress[task]
    
    def update(self, task, advance=1, acc=None):
        self.active_progress[task].update(advance)
        if acc is not None:
            self.active_progress[task].set_postfix_str(f"Best Accuracy: {acc:0.3f}")
    
    def stop_task(self, task):
        self.active_progress[task].close()
        del self.active_progress[task]


def tqdm_print(object):
    tqdm.write(object)
        
    
        
        


