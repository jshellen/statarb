import numpy as np
from statsmodels.tsa.tsatools import lagmat

MAX_EVAL_0 = """2.98    4.13    6.94
                9.47    11.22   15.09
                15.72   17.80   22.25
                21.84   24.16   29.06
                27.92   30.44   35.72
                33.93   36.63   42.23
                39.91   42.77   48.66
                45.89   48.88   55.04
                51.85   54.97   61.35
                57.80   61.03   67.65
                63.73   67.08   73.89
                69.65   73.09   80.12"""

TRACE_0 = """2.98  4.13    6.94
             10.47   12.32   16.36
             21.78   24.28   29.51
             37.03   40.17   46.57
             56.28   60.06   67.64
             79.53   83.94   92.71
             106.74  111.78  121.74
             138.00  143.67  154.80
             173.23  179.52  191.83
             212.47  219.41  232.84
             255.68  263.26  278.00
             302.90  311.13  326.96"""

MAX_EVAL_1 = """7.56    9.16    12.76
                13.91   15.89   20.16
                20.05   22.30   27.07
                26.12   28.59   33.73
                32.17   34.81   40.29
                38.16   40.96   46.75
                44.13   47.07   53.12
                50.11   53.19   59.51
                56.05   59.24   65.79
                61.99   65.30   72.10
                67.93   71.33   78.29
                73.85   77.38   84.51"""

TRACE_1 = """7.56   9.16    12.76
             17.98   20.26   25.08
             32.27   35.19   41.20
             50.53   54.08   61.27
             72.77   76.97   85.34
             99.02   103.84  113.42
             129.23  134.68  145.40
             163.50  169.61  181.51
             201.69  208.45  221.45
             243.96  251.27  265.53
             290.17  298.17  313.75
             340.38  348.99  365.64"""

MAX_EVAL_2 = """2.71    3.84    6.63
                12.30   14.26   18.52
                18.89   21.13   25.86
                25.12   27.58   32.71
                31.24   33.88   39.37
                37.28   40.08   45.87
                43.23   46.23   52.31
                49.29   52.36   58.67
                55.24   58.43   64.99
                61.20   64.51   71.26
                67.13   70.53   77.49
                73.06   76.58   83.70"""

TRACE_2 = """2.71   3.84    6.63
             13.43   15.50   19.94
             27.07   29.80   35.46
             44.49   47.86   54.68
             65.82   69.82   77.82
             91.11   95.75   104.96
             120.37  125.61  135.97
             153.63  159.53  171.09
             190.88  197.37  210.06
             232.11  239.25  253.24
             277.38  285.14  300.29
             326.53  334.98  351.25"""

MAX_EVAL_3 = """10.67   12.52   16.55
                17.23   19.39   23.97
                23.44   25.82   30.83
                29.54   32.12   37.49
                35.58   38.33   44.02
                41.60   44.50   50.47
                47.56   50.59   56.85
                53.55   56.71   63.17
                59.49   62.75   69.44
                65.44   68.81   75.69
                71.36   74.84   81.94
                77.30   80.87   88.11"""

TRACE_3 = """10.67  12.52   16.55
             23.34   25.87   31.16
             39.75   42.91   49.36
             60.09   63.88   71.47
             84.38   88.80   97.60
             112.65  117.71  127.71
             144.87  150.56  161.72
             181.16  187.47  199.81
             221.36  228.31  241.74
             265.63  273.19  287.87
             313.86  322.06  337.97
             366.11  374.91  392.01"""

MAX_EVAL_4 = """2.71    3.84    6.63
                15.00   17.15   21.74
                21.87   24.25   29.26
                28.24   30.82   36.19
                34.42   37.16   42.86
                40.53   43.42   49.41
                46.56   49.58   55.81
                52.58   55.73   62.17
                58.53   61.81   68.50
                64.53   67.90   74.74
                70.46   73.94   81.07
                76.41   79.97   87.23"""

TRACE_4 = """2.71   3.84    6.63
             16.16   18.40   23.15
             32.06   35.01   41.08
             51.65   55.24   62.52
             75.10   79.34   87.78
             102.47  107.34  116.99
             133.79  139.28  150.08
             169.07  175.16  187.20
             208.36  215.12  228.23
             251.63  259.02  273.37
             298.89  306.90  322.41
             350.12  358.72  375.30"""

mapping = {
    "MAX_EVAL_0": MAX_EVAL_0,
    "TRACE_0": TRACE_0,
    "MAX_EVAL_1": MAX_EVAL_1,
    "TRACE_1": TRACE_1,
    "MAX_EVAL_2": MAX_EVAL_2,
    "TRACE_2": TRACE_2,
    "MAX_EVAL_3": MAX_EVAL_3,
    "TRACE_3": TRACE_3,
    "MAX_EVAL_4": MAX_EVAL_4,
    "TRACE_4": TRACE_4
}

class Johansen(object):
    """Implementation of the Johansen test for cointegration.
    References:
        - Hamilton, J. D. (1994) 'Time Series Analysis', Princeton Univ. Press.
        - MacKinnon, Haug, Michelis (1996) 'Numerical distribution functions of
        likelihood ratio tests for cointegration', Queen's University Institute
        for Economic Research Discussion paper.
    """

    def __init__(self, x, model, k=1, trace=True,  significance_level=1):
        """
        :param x: (nobs, m) array of time series. nobs is the number of
        observations, or time stamps, and m is the number of series.
        :param k: The number of lags to use when regressing on the first
        difference of x.
        :param trace: Whether to use the trace or max eigenvalue statistic for
        the hypothesis testing. If False the latter is used.
        :param model: Which of the five cases in Osterwald-Lenum 1992 (or
        MacKinnon 1996) to use.
            - If set to 0, case 0 will be used. This case should be used if
            the input time series have no deterministic terms and all the
            cointegrating relations are expected to have 0 mean.
            - If set to 1, case 1* will be used. This case should be used if
            the input time series has neither a quadratic nor linear trend,
            but may have a constant term, and additionally if the cointegrating
            relations may have nonzero means.
            - If set to 2, case 1 will be used. This case should be used if
            the input time series have linear trends but the cointegrating
            relations are not expected to have linear trends.
            - If set to 3, case 2* will be used. This case should be used if
            the input time series do not have quadratic trends, but they and
            the cointegrating relations may have linear trends.
            - If set to 4, case 2 will be used. This case should be used if
            the input time series have quadratic trends, but the cointegrating
            relations are expected to only have linear trends.
        :param significance_level: Which significance level to use. If set to
        0, 90% significance will be used. If set to 1, 95% will be used. If set
        to 2, 99% will be used.
        """

        self.x = x
        self.k = k
        self.trace = trace
        self.model = model
        self.significance_level = significance_level

        if trace:
            key = "TRACE_{}".format(model)
        else:
            key = "MAX_EVAL_{}".format(model)

        critical_values_str = mapping[key]

        select_critical_values = np.array(
            critical_values_str.split(),
            float).reshape(-1, 3)

        self.critical_values = select_critical_values[:, significance_level]

    def mle(self):
        """Obtain the cointegrating vectors and corresponding eigenvalues.
        Maximum likelihood estimation and reduced rank regression are used to
        obtain the cointegrating vectors and corresponding eigenvalues, as
        outlined in Hamilton 1994.
        :return: The possible cointegrating vectors, i.e. the eigenvectors
        resulting from maximum likelihood estimation and reduced rank
        regression, and the corresponding eigenvalues.
        """

        # Regressions on diffs and levels of x. Get regression residuals.

        # First differences of x.
        x_diff = np.diff(self.x, axis=0)

        # Lags of x_diff.
        x_diff_lags = lagmat(x_diff, self.k, trim='both')

        # First lag of x.
        x_lag = lagmat(self.x, 1, trim='both')

        # Trim x_diff and x_lag so they line up with x_diff_lags.
        x_diff = x_diff[self.k:]
        x_lag = x_lag[self.k:]

        # Include intercept in the regressions if self.model != 0.
        if self.model != 0:
            ones = np.ones((x_diff_lags.shape[0], 1))
            x_diff_lags = np.append(x_diff_lags, ones, axis=1)

        # Include time trend in the regression if self.model = 3 or 4.
        if self.model in (3, 4):
            times = np.asarray(range(x_diff_lags.shape[0])).reshape((-1, 1))
            x_diff_lags = np.append(x_diff_lags, times, axis=1)

        # Residuals of the regressions of x_diff and x_lag on x_diff_lags.
        try:
            inverse = np.linalg.pinv(x_diff_lags)
        except:
            print("Unable to take inverse of x_diff_lags.")
            return None

        u = x_diff - np.dot(x_diff_lags, np.dot(inverse, x_diff))
        v = x_lag - np.dot(x_diff_lags, np.dot(inverse, x_lag))

        # Covariance matrices of the residuals.
        t = x_diff_lags.shape[0]
        Svv = np.dot(v.T, v) / t
        Suu = np.dot(u.T, u) / t
        Suv = np.dot(u.T, v) / t
        Svu = Suv.T

        try:
            Svv_inv = np.linalg.inv(Svv)
        except:
            print("Unable to take inverse of Svv.")
            return None
        try:
            Suu_inv = np.linalg.inv(Suu)
        except:
            print("Unable to take inverse of Suu.")
            return None

        # Eigenvalues and eigenvectors of the product of covariances.
        cov_prod = np.dot(Svv_inv, np.dot(Svu, np.dot(Suu_inv, Suv)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_prod)

        # Normalize the eigenvectors using Cholesky decomposition.
        evec_Svv_evec = np.dot(eigenvectors.T, np.dot(Svv, eigenvectors))
        cholesky_factor = np.linalg.cholesky(evec_Svv_evec)
        try:
            eigenvectors = np.dot(eigenvectors,
                                  np.linalg.inv(cholesky_factor.T))
        except:
            print("Unable to take the inverse of the Cholesky factor.")
            return None

        # Ordering the eigenvalues and eigenvectors from largest to smallest.
        indices_ordered = np.argsort(eigenvalues)
        indices_ordered = np.flipud(indices_ordered)
        eigenvalues = eigenvalues[indices_ordered]
        eigenvectors = eigenvectors[:, indices_ordered]

        return eigenvectors, eigenvalues

    def h_test(self, eigenvalues, r):
        """Carry out hypothesis test.
        The null hypothesis is that there are at most r cointegrating vectors.
        The alternative hypothesis is that there are at most m cointegrating
        vectors, where m is the number of input time series.
        :param eigenvalues: The list of eigenvalues returned from the mle
        function.
        :param r: The number of cointegrating vectors to use in the null
        hypothesis.
        :return: True if the null hypothesis is rejected, False otherwise.
        """

        nobs, m = self.x.shape
        t = nobs - self.k - 1

        if self.trace:
            m = len(eigenvalues)
            statistic = -t * np.sum(np.log(np.ones(m) - eigenvalues)[r:])
        else:
            statistic = -t * np.sum(np.log(1 - eigenvalues[r]))

        critical_value = self.critical_values[m - r - 1]

        if statistic > critical_value:
            return True
        else:
            return False

    def johansen(self):
        """Obtain the possible cointegrating relations and numbers of them.
        See the documentation for methods mle and h_test.
        :return: The possible cointegrating relations, i.e. the eigenvectors
        obtained from maximum likelihood estimation, and the numbers of
        cointegrating relations for which the null hypothesis is rejected.
        """

        nobs, m = self.x.shape

        try:
            eigenvectors, eigenvalues = self.mle()
        except:
            print("Unable to obtain possible cointegrating relations.")
            return None

        rejected_r_values = []
        for r in range(m):
            if self.h_test(eigenvalues, r):
                rejected_r_values.append(r)

        return eigenvectors, rejected_r_values