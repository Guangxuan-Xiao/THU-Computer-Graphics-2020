# Computer Graphics PA3: Drawing Basis Function.
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


class Bernstein:
    """
    Bernstein basis computer using Cox De-Boor algorithm
    - The special case of De Casteljau is also considered
    """

    def __init__(self, n, k, t: np.ndarray):
        """
        Precompute a Bernstein Basis function
        :param n: number of control points
        :param k: number of degree
        :param t: knot vector
        """
        assert t.shape[0] == n + k + 1
        self.n = n
        self.k = k
        self.t = t
        self.tpad = np.pad(self.t, (0, self.k), 'constant',
                           constant_values=(0., self.t[-1]))
        print("TPAD", self.tpad)

    @staticmethod
    def bezier_knot(k):
        # degree k must have n control points.
        n = k + 1
        return np.concatenate((np.zeros(n), np.ones(n)))

    def get_bpos(self, mu):
        if not mu >= self.t[0] or not mu <= self.t[-1]:
            raise ValueError
        # lower bound
        if mu == self.t[0]:
            bpos = np.searchsorted(self.t, mu, 'right') - 1
            print("kjhkjh", mu, bpos)
        else:
            bpos = max(0, np.searchsorted(self.t, mu) - 1)
        return bpos

    def get_valid_range(self):
        start_t = self.t[self.k]
        end_t = self.t[-self.k-1]
        return start_t, end_t

    def evaluate(self, mu):
        bpos = self.get_bpos(mu)
        print("bpos: ", bpos)
        s = np.zeros(self.k + 2)
        s[-2] = 1
        ds = np.ones(self.k + 1)
        for p in range(1, self.k + 1):
            for ii in range(self.k - p, self.k + 1):
                i = ii + bpos - self.k
                if self.tpad[i + p] == self.tpad[i]:
                    w1 = mu
                    dw1 = 1.
                else:
                    w1 = (mu - self.tpad[i]) / \
                        (self.tpad[i + p] - self.tpad[i])
                    dw1 = 1 / (self.tpad[i + p] - self.tpad[i])
                if self.tpad[i + p + 1] == self.tpad[i + 1]:
                    w2 = 1 - mu
                    dw2 = -1.
                else:
                    w2 = (self.tpad[i + p + 1] - mu) / \
                        (self.tpad[i + p + 1] - self.tpad[i + 1])
                    dw2 = - 1 / (self.tpad[i + p + 1] - self.tpad[i + 1])
                if p == self.k:
                    ds[ii] = (dw1 * s[ii] + dw2 * s[ii + 1]) * p
                s[ii] = w1 * s[ii] + w2 * s[ii + 1]

        s = s[:-1]
        lsk = bpos - self.k
        rsk = self.n - bpos - 1
        if lsk < 0:
            s = s[-lsk:]
            ds = ds[-lsk:]
            lsk = 0
        if rsk < 0:
            print("Rsk:", rsk, len(s))
            s = s[:rsk]
            ds = ds[:rsk]

        return s, ds, lsk


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Basis function visualizer for B-Spline')
    parser.add_argument('--N', type=int, default=4,
                        help='Number of control points')
    parser.add_argument('--k', type=int, default=3, help='Spline order x^k')
    parser.add_argument('--bspline', action='store_true',
                        help='Draw B-Spline instead of Bezier')
    parser.add_argument('--valid', action='store_true',
                        help='Draw only valid range of parameter')
    args = parser.parse_args()

    n = args.N
    k = args.k
    if args.bspline:
        knot = np.linspace(0., 1., n + k + 1)
        print(knot)
        description = "B-Spline (N = %d, k = %d)" % (n, k)
    else:
        if n != k + 1:
            print("Error: For bezier curve N == k + 1")
            sys.exit(0)
        knot = Bernstein.bezier_knot(k)
        print(knot)
        description = "Bezier (N = %d)" % n

    b = Bernstein(n, k, knot)
    if args.valid:
        t_range = np.linspace(b.get_valid_range()[
                              0], b.get_valid_range()[1], 5)
    else:
        t_range = np.linspace(0., 1., 5)

    lines = []
    deriv_lines = []
    for t in t_range:
        pt, dpt, lsk = b.evaluate(t)
        print("pt:", pt)
        print("dpt:", dpt)
        print("lsk:", lsk)
        expanded_pt = np.zeros(n,)
        expanded_pt[lsk:lsk+pt.shape[0]] = pt
        expanded_dpt = np.zeros(n,)
        expanded_dpt[lsk:lsk+dpt.shape[0]] = dpt
        lines.append(expanded_pt)
        deriv_lines.append(expanded_dpt)
        print("T: ", t)
    lines = np.asarray(lines).T
    deriv_lines = np.asarray(deriv_lines).T

    plt.figure()
    for l in lines:
        plt.plot(t_range, l)
    plt.grid()
    plt.title('Basis function for ' + description)

    plt.figure()
    for dl in deriv_lines:
        plt.plot(t_range, dl)
    plt.grid()
    plt.title('Derivative for ' + description)
    plt.show()
