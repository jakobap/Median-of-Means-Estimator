import numpy as np
import matplotlib.pyplot as plt
import random


class MeanEstimation():
    def __init__(self, n_pop, n_sample, err_thresh, n_blocks, rmom_n, population):
        self.n_pop = n_pop  # Total Number of observations in population
        self.n_sample = n_sample  # Number of observations sampled from population
        self.err_tresh = err_thresh  # Error Threshold for performance evaluation
        self.population = population  # RV Population
        self.sample = None  # Estimator evaluation sample
        self.population_mean = None  # Ground Truth Benchmark
        self.sample_mean = None  # Estimation
        self.n_blocks = n_blocks  # Number of Blocks for and rmom
        self.rmom_n = rmom_n  # Number of mom to average over for rmom
        self.eval_rounds = 100  # Number of independent samples estimators are tested against
        self.em_error = None  # P(mn - m) > e for empirical mean
        self.mom_error = None  # P(mn - m) > e for simple median of means
        self.rmom_error = None  # P(mn - m) > e for robust median of means
        self.tm_error = None  # P(mn - m) > e for trimmed mean

    def _emp_mean(self, seq):
        return np.sum(seq) / len(seq)

    def _median_of_means(self, seq):
        """
        1. If clause to prevent n_blocks > n_observations
        2. Dividing data in n_blocks random blocks by generating numbers 0 to n_blocks obs per block times & shuffle
        3. List comp to iterate blocks, in each iteration we index from the total sequence the items that have been
            selected for class i, then we compute the empirical mean over these
        4. We return the median over the list of means
        """
        if self.n_blocks > len(seq):  # preventing the n_blocks > n_observations
            self.n_blocks = int(np.ceil(len(seq) / 2))

        # dividing seq in k random blocks
        indic = np.array(list(range(self.n_blocks)) * int(len(seq) / self.n_blocks))
        np.random.shuffle(indic)

        # computing and saving mean per block
        means = [self._emp_mean(seq[list(np.where(indic == block)[0])]) for block in range(self.n_blocks)]

        # return median
        return np.median(means)

    def _rob_median_of_means(self, seq):
        res = [self._median_of_means(seq) for _ in range(self.rmom_n)]
        return self._emp_mean(res)

    def _trimmed_mean(self, seq):
        upperb = np.mean(seq) + np.std(seq)
        lowerb = np.mean(seq) - np.std(seq)
        seq = seq[seq <= upperb]
        seq = seq[seq >= lowerb]
        return np.mean(seq)

    def _new_sample(self):
        self.sample = np.array(random.choices(self.population, k=self.n_sample))

    def _residual(self):
        return round(np.abs(self.population_mean - self.sample_mean), 4)

    def _estimator_benchmark(self, estimator):
        ## Benchmarking Empirical Mean
        resid = []
        for _ in range(self.eval_rounds):
            self._new_sample()
            self.population_mean = estimator(self.population)
            self.sample_mean = estimator(self.sample)
            resid.append(self._residual())
        return len(np.where(np.array(resid) >= self.err_tresh)[0]) / len(resid)

    def mean_estimation(self):
        self.em_error = self._estimator_benchmark(self._emp_mean)
        self.mom_error = self._estimator_benchmark(self._median_of_means)
        self.rmom_error = self._estimator_benchmark(self._rob_median_of_means)
        self.tm_error = self._estimator_benchmark(self._trimmed_mean)
        return [self.em_error, self.mom_error, self.rmom_error, self.tm_error]


def run_complete_benchmark(n_list, error_thresh, student_df, normal_sd):
    result_list_normal = [MeanEstimation(n_pop=n, n_sample=int(n * .5), err_thresh=.05, rmom_n=10, n_blocks=20,
                                         population=np.random.normal(loc=0, size=n, scale=1)).mean_estimation()
                          for n in n_list]

    result_list_student = [MeanEstimation(n_pop=n, n_sample=int(n * .5), err_thresh=.05, rmom_n=10, n_blocks=20,
                                         population=np.random.standard_t(df=student_df, size=n)).mean_estimation()
                          for n in n_list]

    plt.title(f'Estimator Error: err_thresh {error_thresh}, student_df {student_df}, norm_sd {normal_sd}')
    plt.plot(n_list, np.array(result_list_normal)[:, 0], marker='+', label='EM Normal', alpha=.5, c='green')
    plt.plot(n_list, np.array(result_list_normal)[:, 1], marker='x', label='MoM Normal', alpha=.5, c='green')
    plt.plot(n_list, np.array(result_list_normal)[:, 2], marker='o', label='RMoM Normal', alpha=.5, c='green')
    plt.plot(n_list, np.array(result_list_normal)[:, 3], marker='.', label='TM Normal', alpha=.5, c='green')
    plt.plot(n_list, np.array(result_list_student)[:, 0], marker='+', label='EM Student', alpha=.5, c='red')
    plt.plot(n_list, np.array(result_list_student)[:, 1], marker='x', label='MoM Student', alpha=.5, c='red')
    plt.plot(n_list, np.array(result_list_student)[:, 2], marker='o', label='RMoM Student', alpha=.5, c='red')
    plt.plot(n_list, np.array(result_list_student)[:, 3], marker='.', label='TM Student', alpha=.5, c='red')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# For Debugging
if __name__ == '__main__':
    run_complete_benchmark(n_list=[5, 10, 50, 200, 500, 1000, 2000, 5000, 10000], error_thresh=.1,
                           student_df=1, normal_sd=1)
    run_complete_benchmark(n_list=[5, 10, 50, 200, 500, 1000, 2000, 5000, 10000], error_thresh=.1,
                           student_df=3, normal_sd=1)
    print('hello world')
