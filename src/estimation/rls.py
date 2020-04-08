import numpy as np
from collections import deque


def system_identification_setup(F):
  '''
  F should specify an instance of a recursive filter implementing ff
  and fb.  We return the filter parameters of the filter w, which can
  be used to identify a model of a system.
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return F.w
  ff_fb.x_hat_prev = 0
  return ff_fb


def one_step_pred_setup(F):
  '''
  F should specify an instance of a recursive filter implementing ff
  and fb.  This function uses F to make a 1 step prediction of a
  process.  This is basically a method to track the process over time.

  Note that there is an implicit prediction of 0 for x(0)
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return x_hat
  ff_fb.x_hat_prev = 0
  return ff_fb


def equalizer_setup(F, rx_delay = 0):
  '''
  Sets up a function to use F as an adaptive equalizer.  The
  argument x to ff_fb is the filter's observation, and d is the true
  value of what it should estimate.  Hence if we have x = Hd + v where
  Hd is a filtered data sequence and v is noise, then passing in x and
  d to ff_fb will train an equalizer for H.

  If d is None, we use decision feedback.

  The rx_delay parameter specifies the delay that will be added at the
  receiver.  This is to offset the delay of the channel.  It should be
  long enough that all the "useful" information for predicting d(n -
  rx_delay) from the input sequence x has made it's way into the
  filter We then delay the d inputs by this amount so that the filter
  does not need to be non-causal.

  Note that the filter must have enough taps to accomodate this delay.

  '''
  assert F.p >= rx_delay, 'Filter does not have enough taps'
  def ff_fb(x, d = None):
    d_hat = F.ff(x)
    if d: #Training mode
      ff_fb.D.appendleft(d)
    else:
      ff_fb.D.appendleft(0)


    if ff_fb.D[-1]: #Check for training data
      F.fb(ff_fb.D.pop() - d_hat)
    else: #Else, decision directed feedback
      d = 1 if d_hat >= 0 else -1
      F.fb(d - d_hat)

    return d

  ff_fb.D = deque([0]*(rx_delay + 1),
                  maxlen = rx_delay + 1)
  return ff_fb


class RLSFilter2(object):
    '''
    Basic Recursive Least Squares adaptive filter. Suppose we wish to
    estimate a signal d(n) from a related signal x(n).  (for example, we
    could have noisy measurements x(n) = d(n) + v(n)) We make a linear
    estimate d^(n) (d hat of n).

    d^(n) = sum(w(k)*x(n - k) for k in range(p + 1))

    where x(n) denotes the measurements, and w(k) are the coefficients
    of our filter.

    Assuming that we can observe the true value of d(n), at least for
    some n, then we can update our filter coefficents w(k) to attempt to
    more accurately track d(n).  The RLS algorithm minimizes the
    exponentially weighted squared error over the last p samles.  In
    order to update the filter coefficients pass in the error on the
    previous prediction to LMS.fb.

    w(k) <- w(k) + mu*e(n)*x(n - k)^*
    '''

    def __init__(self, p, lmbda=0.9, delta=1e-3):
        '''
        p denotes the order of the filter.  A zero order filter makes a
        prediction of d(n) based only on x(n), and a p order filter uses
        x(n) ... x(n - p).

        The 0 <= lmbda <= 1 paramter specifies the exponential weighting
        of the errors.  As lmbda decreases, the data is "forgotten" more quickly.

        E(n) = sum_i[lmbda**(n - i) * (d(i) - d^(i))**2]

        delta specifies the parameter for the initial covariance matrix
        P(0) = delta*I
        '''
        if not isinstance(p, int):
            raise TypeError('p has to be type <int>')

        if p <= 0:
            raise ValueError('Filter order must be non-negative.')

        if not isinstance(lmbda, (int, float)):
            raise TypeError('Forgetting factor must be type <int> or <float>.')

        if lmbda > 1:
            raise ValueError('Forgetting factor must be in (0, 1].')

        if lmbda < 0:
            raise ValueError('Forgetting factor must be in (0, 1].')

        if not isinstance(delta, (int, float)):
            raise TypeError('Initial covariance must be type of <int> or <float>.')

        if delta < 0:
            raise ValueError('Initial covariance must be positive definite.')

        self.p = p  # Filter order
        self.lmbda = float(lmbda)  # Forgetting factor
        delta = float(delta)
        self.Rt = delta * np.eye(p + 1)  # Empirical covariance
        self.Rt_inv = (1 / delta) * np.eye(p + 1)  # Inverse of Rt
        self.x = deque([0] * (p + 1), maxlen=p + 1)  # Saved data vector
        self.x_hat_prev = 0
        self.d = []
        self.w = np.array([0] * (p + 1))
        self.s = []

    def ff(self, x_n):
        '''
        Feedforward.  Make a new prediction of d(n) based on the new input
        x(n)
        '''

        if not isinstance(x_n, (int, float)):
            raise TypeError('Input should be type of <float>.')

        self.x.appendleft(x_n)

        return sum([wi * xi for (wi, xi) in zip(self.w, self.x)])

    def fb(self, e):
        '''
        Feedback.  Updates the coefficient vector w based on an error
        feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
        the previous prediction.
        '''
        if not isinstance(e, (int, float)):
            raise TypeError('Input should be type of <float>.')

        l = (1. / self.lmbda)
        x = np.array(self.x).reshape(self.p + 1, 1)  # Make a column vector
        w = self.w.reshape(self.p + 1, 1)  # Make a column vector

        # ---------------------------
        # This is more stable
        # But, it completely defeats the purpose of updating Rt_inv...
        # self.Rt = self.lmbda*self.Rt + np.dot(x, x.T)
        # g = sp.linalg.solve(self.Rt, x, sym_pos = True, check_finite = False)

        # ---------------------------
        u = np.dot(self.Rt_inv, x)  # Intermediate value
        d = self.lmbda + np.dot(x.T, u)
        self.d.append(d[0][0])
        g = u / d  # Gain vector
        self.Rt_inv = l * (self.Rt_inv - np.dot(g, u.T))
        w_old = self.w.copy()
        self.w = (w + e * g).reshape(self.p + 1)  # Update the filter
        self.Rt_inv = l * np.dot((np.eye(self.p + 1) - np.dot(g, x.T)),
                                 self.Rt_inv)

        self.Rt_inv = 0.5 * (self.Rt_inv + self.Rt_inv.T)  # Ensure symmetry

        n = len(self.s)
        self.s.append(sum((self.w - w_old)**2))


        return

    def ff_fb(self, x):
        """
        Forward backward algorithm. Updates the coefficient vector w
        based on new dta.
        """

        # Predict new observation
        self.fb(x - self.x_hat_prev)

        # Update parameters
        x_hat = self.ff(x)

        # Save previous
        self.x_hat_prev = x_hat

        return self.w.copy()