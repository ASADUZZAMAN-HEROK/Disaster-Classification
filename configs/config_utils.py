import tabulate

from configs.config import Config


def tree_print(tree, level=0, prev_level=[], is_last_item=False):
    def getPrefix(level, prev_level=[], is_last_item=False):
        while len(prev_level) < level + 1:
            prev_level.append(False)
        prev_level[level] = is_last_item
        prefix = ""
        for i in range(level):
            if prev_level[i]:
                prefix = prefix + "   "
            else:
                prefix = prefix + "|  "
        if is_last_item:
            prefix += "└── "
        else:
            prefix += "├── "
        return prefix

    if isinstance(tree, dict):
        i = 0
        len_ = len(tree)
        return_str = ""
        for k, v in tree.items():
            i += 1
            prefix = getPrefix(level, prev_level, i == len_)
            return_str += f"{prefix}{k}"
            if not isinstance(v, dict):
                is_last_item = True
            else:
                return_str += "\n"
            return_str += tree_print(v, level + 1, prev_level, is_last_item)
        return return_str
    else:
        return ": " + (str(tree) if not isinstance(tree, str) else tree) + "\n"


def show_config(cfg: Config):
    table = tabulate.tabulate(
        {"": ["Configuration\n" + tree_print(cfg.to_dict())]},
        tablefmt="grid",
    )
    print(table)
