import java.io.*;
import java.util.*;

public class Main {
    static final long INF = 1_000_000_000_000_000_000L + 5;

    static long ceilDiv(long x, long y) {
        return (x + y - 1) / y;
    }

    static long capMul(long a, long b) {
        if (a >= INF || b >= INF) return INF;
        if (a > INF / b) return INF;
        return a * b;
    }

    static class SegTree {
        int n, size;
        long[] seg;
        SegTree(long[] arr) {
            this.n = arr.length - 1;
            size = 1;
            while (size < n) size <<= 1;
            seg = new long[2 * size];
            Arrays.fill(seg, 1L);
            for (int i = 1; i <= n; i++) seg[size + i - 1] = arr[i];
            for (int i = size - 1; i >= 1; i--) seg[i] = capMul(seg[2*i], seg[2*i+1]);
        }
        void set(int idx, long val) {
            int i = size + idx - 1;
            seg[i] = val;
            i >>= 1;
            while (i >= 1) {
                seg[i] = capMul(seg[2*i], seg[2*i+1]);
                i >>= 1;
            }
        }
        long allProd() { return seg[1]; }
    }

    public static void main(String[] args) throws Exception {
        FastScanner fs = new FastScanner(System.in);
        int N = fs.nextInt();
        long[] a = new long[N + 1];
        for (int i = 1; i <= N; i++) a[i] = fs.nextLong();

        long[] b = new long[N + 1];
        b[1] = a[1] + 1;
        for (int i = 2; i <= N; i++) b[i] = ceilDiv(a[i], a[i-1]);

        SegTree st = new SegTree(b);

        int Q = fs.nextInt();
        StringBuilder out = new StringBuilder();

        for (int qi = 0; qi < Q; qi++) {
            int idx = fs.nextInt();
            long v = fs.nextLong();
            long t = fs.nextLong();

            a[idx] = v;

            if (idx == 1) {
                b[1] = a[1] + 1;
                st.set(1, b[1]);
                if (N >= 2) {
                    b[2] = ceilDiv(a[2], a[1]);
                    st.set(2, b[2]);
                }
            } else {
                b[idx] = ceilDiv(a[idx], a[idx-1]);
                st.set(idx, b[idx]);
                if (idx < N) {
                    b[idx+1] = ceilDiv(a[idx+1], a[idx]);
                    st.set(idx+1, b[idx+1]);
                }
            }

            long p = st.allProd();          // capped
            long first = (p <= INF) ? INF : p + (N - 1);
            if (first > t) {
                out.append(0).append('\n');
            } else {
                long cnt = 1 + (t - first) / p;
                out.append(cnt * a[N]).append('\n');
            }
        }

        System.out.print(out.toString());
    }

    static class FastScanner {
        private final InputStream in;
        private final byte[] buffer = new byte[1 << 16];
        private int ptr = 0, len = 0;
        FastScanner(InputStream is) { in = is; }
        private int read() throws IOException {
            if (ptr >= len) {
                len = in.read(buffer);
                ptr = 0;
                if (len <= 0) return -1;
            }
            return buffer[ptr++];
        }
        long nextLong() throws IOException {
            int c;
            do c = read(); while (c <= ' ' && c != -1);
            long sgn = 1;
            if (c == '-') { sgn = -1; c = read(); }
            long x = 0;
            while (c > ' ') {
                x = x * 10 + (c - '0');
                c = read();
            }
            return x * sgn;
        }
        int nextInt() throws IOException { return (int) nextLong(); }
    }
}