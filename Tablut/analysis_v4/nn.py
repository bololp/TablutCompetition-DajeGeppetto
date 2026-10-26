import numpy as np, time
LEAK = 0.01
def lk(x): return np.where(x > 0, x, LEAK*x)
def dlk(x): return np.where(x > 0, 1.0, LEAK)

class Net:
    """tronco 24 leaky (condiviso) + 2 head (chi muove): 8 leaky -> 1 tanh. Stessa struttura dell'MLP del progetto (B)."""
    def __init__(self, nin, seed, trunk=24, head=8):
        r = np.random.RandomState(seed)
        self.p = {
            'W1': r.randn(nin, trunk) * np.sqrt(2.0/nin), 'b1': np.zeros(trunk),
            'W2': r.randn(2, trunk, head) * np.sqrt(2.0/trunk), 'b2': np.zeros((2, head)),
            'W3': r.randn(2, head, 1) * np.sqrt(2.0/head) * 0.2, 'b3': np.zeros((2, 1)),
        }
    def forward(self, X, role, keep=False):
        p = self.p
        z1 = X @ p['W1'] + p['b1']; h1 = lk(z1)
        out = np.zeros(len(X)); cache = [z1, h1, {}]
        for r in (0, 1):
            m = role == r
            if not m.any(): continue
            z2 = h1[m] @ p['W2'][r] + p['b2'][r]; h2 = lk(z2)
            z3 = h2 @ p['W3'][r] + p['b3'][r]; y = np.tanh(z3)[:, 0]
            out[m] = y; cache[2][r] = (m, z2, h2, y)
        return (out, cache) if keep else out
    def grads(self, X, role, t):
        out, (z1, h1, hc) = self.forward(X, role, True)
        n = len(X); g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh1 = np.zeros_like(h1)
        for r, (m, z2, h2, y) in hc.items():
            dy = 2*(y - t[m])/n
            dz3 = (dy*(1 - y*y))[:, None]
            g['W3'][r] = h2.T @ dz3; g['b3'][r] = dz3.sum(0)
            dh2 = dz3 @ self.p['W3'][r].T
            dz2 = dh2 * dlk(z2)
            g['W2'][r] = h1[m].T @ dz2; g['b2'][r] = dz2.sum(0)
            dh1[m] = dz2 @ self.p['W2'][r].T
        dz1 = dh1 * dlk(z1)
        g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
        return g, float(np.mean((out - t)**2))

def train(Xtr, rtr, ytr, Xte, rte, yte, seed, epochs=12, batch=256, lr0=3e-3):
    net = Net(Xtr.shape[1], seed)
    m = {k: np.zeros_like(v) for k, v in net.p.items()}; v = {k: np.zeros_like(x) for k, x in net.p.items()}
    rs = np.random.RandomState(seed + 1000); n = len(Xtr); steps = epochs * ((n + batch - 1)//batch); s = 0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n, batch):
            ix = perm[i:i+batch]
            g, _ = net.grads(Xtr[ix], rtr[ix], ytr[ix])
            s += 1; lr = lr0 * 0.5*(1 + np.cos(np.pi * s/steps))
            for k in net.p:
                m[k] = b1*m[k] + (1-b1)*g[k]; v[k] = b2*v[k] + (1-b2)*g[k]**2
                net.p[k] -= lr * (m[k]/(1-b1**s)) / (np.sqrt(v[k]/(1-b2**s)) + eps)
    pred = net.forward(Xte, rte)
    return net, float(np.mean((pred - yte)**2)), pred

if __name__ == '__main__':
    # controllo del gradiente con differenze finite (double)
    r = np.random.RandomState(0)
    X = r.randn(40, 7); role = r.randint(0, 2, 40); t = r.choice([-1., 0., 1.], 40)
    net = Net(7, 1)
    g, _ = net.grads(X, role, t)
    worst = 0; cnt = 0
    for k in net.p:
        flat = net.p[k].reshape(-1); gf = g[k].reshape(-1)
        for i in range(len(flat)):
            o = flat[i]; h = 1e-6
            flat[i] = o + h; lp = np.mean((net.forward(X, role) - t)**2)
            flat[i] = o - h; lm = np.mean((net.forward(X, role) - t)**2)
            flat[i] = o
            num = (lp - lm)/(2*h); err = abs(num - gf[i]) / max(1e-7, abs(num) + abs(gf[i]))
            worst = max(worst, err); cnt += 1
    print('gradient check: parametri', cnt, 'errore relativo massimo', worst)
