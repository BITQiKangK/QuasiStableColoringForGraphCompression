import torch


class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def reset(self, run):
        assert run >= 0 and run < len(self.results)
        self.results[run] = []

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            s = ""
            s += f'Run {run + 1:02d}:\n'
            s += f'Highest Train: {result[:, 0].max():.4f}\n'
            s += f'Highest Valid: {result[:, 1].max():.4f}\n'
            s += f'  Final Train: {result[argmax, 0]:.4f}\n'
            s += f'   Final Test: {result[argmax, 2]:.4f}\n'
            return s
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            s = ""
            s += f'All runs:'
            r = best_result[:, 0]
            s += f'Highest Train: {r.mean():.4f} ± {r.std():.4f}'
            r = best_result[:, 1]
            s += f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}'
            r = best_result[:, 2]
            s += f'  Final Train: {r.mean():.4f} ± {r.std():.4f}'
            r = best_result[:, 3]
            s += f'   Final Test: {r.mean():.4f} ± {r.std():.4f}'
