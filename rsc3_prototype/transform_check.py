"""
Fast MDCT/IMDCT, decoder-grade: analysis = fold + fast DCT-IV, synthesis = fast DCT-IV +
transpose of the fold. Both verified against the matrix kernel, so the whole transform is
fold + one length-(N/2) complex FFT + twiddles, which is what makes it Luau-portable.
"""
import numpy as np

def dct4(v):
    """DCT-IV via one length-L complex inverse DFT (half-bin shifted), read even/odd."""
    L = len(v); m = np.arange(L)
    q = v * np.exp(1j * np.pi * m / (2 * L))
    A = np.fft.ifft(q) * L
    B = np.fft.ifft(q * np.exp(1j * np.pi * m / L)) * L
    C = np.empty(L)
    r = np.arange(L // 2)
    ke, ko = 2 * r, 2 * r + 1
    C[ke] = 2 * (np.exp(1j * np.pi * (ke + 0.5) / (2 * L)) * A[r]).real
    C[ko] = 2 * (np.exp(1j * np.pi * (ko + 0.5) / (2 * L)) * B[r]).real
    return C

def mdct(x, N2):
    """One windowed block of N2 samples -> N2/2 coefficients (validated fold)."""
    h = N2 // 4
    c, d = x[2 * h:3 * h], x[3 * h:]
    a, b = x[:h], x[h:2 * h]
    return 0.5 * dct4(np.concatenate([-c[::-1] - d, a - b[::-1]]))

def imdct(X, N2):
    """N2/2 coefficients -> N2 samples. The fold's transpose, scale 1/(N2/2)."""
    v = dct4(X)
    p, q = v[:N2 // 4], v[N2 // 4:N2 // 2]
    return (2.0 / N2) * np.concatenate([q, -q[::-1], -p[::-1], -p])

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    for N2 in (32, 128, 512, 2048, 4096):
        n = np.arange(N2); k = np.arange(N2 // 2)
        C = np.cos(np.pi / (N2 / 2) * (n + 0.5 + N2 / 4)[:, None] * (k + 0.5)[None, :])
        x = rng.standard_normal(N2)
        ea = np.max(np.abs(mdct(x, N2) - C.T @ x))
        es = np.max(np.abs(imdct(mdct(x, N2), N2) - (2.0 / (N2 / 2)) * (C @ mdct(x, N2) / 1)))
        # windowed OLA over a whole signal, both sides fast
        w = np.sin(np.pi * (np.arange(N2) + 0.5) / N2)
        H = N2 // 2
        sig = rng.standard_normal(64 * H + N2)
        out = np.zeros(len(sig) + 2 * N2)
        nf = int(np.ceil(len(sig) / H)) + 1
        for i in range(nf):
            s = min(i * H, len(sig))
            blk = np.concatenate([sig[s:min(s + N2, len(sig))],
                                  np.zeros(max(0, s + N2 - len(sig)))])[:N2]
            if blk.shape[0] != N2:
                blk = np.concatenate([blk, np.zeros(N2 - blk.shape[0])])
            out[i * H:i * H + N2] += w * imdct(mdct(w * blk, N2), N2)
        e = np.max(np.abs(out[N2:len(sig)] - sig[N2:len(sig)]))
        print(f"N2={N2:5d} analysis {ea:.2e}  synthesis {es:.2e}  OLA {e:.2e}   "
              f"({(N2//2)//2}-point complex FFT per block)")
