from tqdm.auto import tqdm 

class ProgressBars:
    def __init__(self, desc="Processing", mininterval=0.01, ncols=100, leave=True):
        self.desc = desc
        self.mininterval = mininterval
        self.ncols = ncols
        self.leave = leave
        self.active_progress = {}

    def add_task(self, task, desc, total, completed=0, acc=None):
        if task not in self.active_progress:
            self.active_progress[task] = tqdm(
                desc=desc,
                total=total,
                mininterval=self.mininterval,
                ncols=self.ncols,
                leave=self.leave,
                dynamic_ncols=True
            )
        else:
            self.active_progress[task].set_description(desc)
            self.active_progress[task] = self.active_progress[task]
            self.active_progress[task].total = total
            self.active_progress[task].n = 0
            self.active_progress[task].refresh()

        self.active_progress[task].n = completed
        if acc is not None:
            self.active_progress[task].set_postfix_str(f"Best Accuracy: {acc:.3f}")
        self.active_progress[task].refresh()
        return task

    def remove_task(self, task):
        if task in self.active_progress:
            self.active_progress[task].close()
            del self.active_progress[task]

    def update(self, task, advance=1, acc=None):
        if task in self.active_progress:
            self.active_progress[task].update(advance)
            if acc is not None:
                self.active_progress[task].set_postfix_str(f"Best Accuracy: {acc:.3f}")
            self.active_progress[task].refresh()

    def stop_task(self, task):
        self.remove_task(task)

def tqdm_print(message):
    tqdm.write(message)
