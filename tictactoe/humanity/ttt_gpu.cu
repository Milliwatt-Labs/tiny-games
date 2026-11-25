// Tic-tac-toe batch player for the "more games than humanity" run (tictactoe/humanity).
//
// One thread plays whole games in registers. The net move is the integer forward pass of net.py: canonical
// symmetry (game.canonical: first of the 8 SYMS in order with the lexicographically smallest image), side-relative
// encoding (game.encode), h = act(W1 x + b1), z = W2 h + b2, argmax over the empty cells with ties to the lowest
// index in the net's frame, mapped back through the symmetry. Ternary rows are stored as +1 / -1 bit masks, so a
// hidden unit is four popcounts. Tables (SYMS, optimal-move masks, minimax values) come from game.py via
// humanity.py; nothing here decides what a move is worth.
//
// RNG: Philox4x32-10, counter = (game index lo, game index hi, arm id, block), key = (seed0, seed1). Each game has
// its own counter stream, so a count depends only on (arm, seed, game range), not on the launch shape.
//
// Build: nvcc -O3 -arch=native -o ttt_gpu ttt_gpu.cu        (GPU)
//        c++ -O2 -std=c++17 -x c++ -o ttt_cpu ttt_gpu.cu    (CPU, same functions; for tests)
//
// Usage:
//   ttt_gpu parity <tables.bin> <net.bin> <codes.txt>           net move for each base-3 board code, one per line
//   ttt_gpu trace  <tables.bin> <net.bin> ARM start n          per game: outcome theory plies seq
//   ttt_gpu run    <tables.bin> <net.bin> ARM start n bitmap.out   counts as one JSON line, seen-games bitmap
//   ARM = side opp thresh open_k seed0 seed1 arm_id
//     side 1 = net plays X, 2 = net plays O, 0 = net plays both (after open_k random moves)
//     opp 0 perfect (uniform over value-keeping moves), 1 uniform random, 2 kid, 3 mix; thresh = p * 2^32
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#ifdef __CUDACC__
#define HD __device__ __forceinline__
#define CONST __constant__
#define POPC(v) __popc(v)
#define MULHI(a, b) __umulhi(a, b)
#define CTZ(v) (__ffs(v) - 1)
#else
#define HD inline
#define CONST static
#define POPC(v) __builtin_popcount(v)
#define MULHI(a, b) ((uint32_t)(((uint64_t)(a) * (uint64_t)(b)) >> 32))
#define CTZ(v) __builtin_ctz(v)
#endif

#define MAXH 32
#define NCODES 19683
#define SEQ_TOTAL 982800  // sum over L = 5..9 of 9!/(9-L)!
#define BITMAP_WORDS ((SEQ_TOTAL + 31) / 32)

struct Net {
    int H, planes, step, sym;
    uint32_t pos[MAXH], neg[MAXH];
    int b1[MAXH];
    int w2[9][MAXH];
    int b2[9];
};

CONST Net c_net;
CONST uint8_t c_syms[8][9];
CONST uint32_t c_seq_off[10];


struct Tables {
    const uint16_t* opt;  // optimal-move mask by base-3 code (decision positions only)
    const int8_t* valx;   // minimax value from X's view (+1 X wins, 0 draw, -1 O wins)
};

struct Arm {
    int side, opp, open_k;
    uint32_t thresh, seed0, seed1, arm_id;
};

// ---- Philox4x32-10 ------------------------------------------------------------------------------------------
struct Rng {
    uint32_t c[4], k[2], buf[4];
    int idx;
};

HD void philox(const uint32_t* ctr, const uint32_t* key, uint32_t* out) {
    uint32_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3], k0 = key[0], k1 = key[1];
    for (int r = 0; r < 10; r++) {
        uint32_t hi0 = MULHI(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint32_t hi1 = MULHI(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        uint32_t n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

HD void rng_init(Rng& g, uint64_t game, const Arm& a) {
    g.c[0] = (uint32_t)game; g.c[1] = (uint32_t)(game >> 32); g.c[2] = a.arm_id; g.c[3] = 0;
    g.k[0] = a.seed0; g.k[1] = a.seed1; g.idx = 4;
}

HD uint32_t rng_next(Rng& g) {
    if (g.idx == 4) { philox(g.c, g.k, g.buf); g.c[3]++; g.idx = 0; }
    return g.buf[g.idx++];
}

// the r-th set bit (ascending) of mask, r = floor(u * popcount / 2^32)
HD int pick(uint32_t mask, uint32_t u) {
    uint32_t r = MULHI(u, (uint32_t)POPC(mask));
    for (uint32_t i = 0; i < r; i++) mask &= mask - 1;
    return CTZ(mask);
}

// ---- rules ----------------------------------------------------------------------------------------------------
HD bool won(uint32_t m) {
    return ((m & 0x7) == 0x7) | ((m & 0x38) == 0x38) | ((m & 0x1C0) == 0x1C0) | ((m & 0x49) == 0x49) |
           ((m & 0x92) == 0x92) | ((m & 0x124) == 0x124) | ((m & 0x111) == 0x111) | ((m & 0x54) == 0x54);
}

HD int code3(uint32_t x, uint32_t o) {
    int c = 0;
    for (int i = 8; i >= 0; i--) c = c * 3 + ((x >> i) & 1) + 2 * ((o >> i) & 1);
    return c;
}

// empty cells that complete a line for `mine`
HD uint32_t completing(uint32_t mine, uint32_t empty) {
    const uint32_t LINES[8] = {0x7, 0x38, 0x1C0, 0x49, 0x92, 0x124, 0x111, 0x54};
    uint32_t out = 0;
    for (int l = 0; l < 8; l++) {
        uint32_t L = LINES[l];
        if (POPC(L & mine) == 2 && (L & empty)) out |= L & empty;
    }
    return out;
}

// ---- the net --------------------------------------------------------------------------------------------------
HD int net_move(const Net& n, const uint8_t (*syms)[9], uint32_t x, uint32_t o) {
    bool xmove = POPC(x) == POPC(o);
    uint32_t me = xmove ? x : o, op = xmove ? o : x;
    int pi = 0;
    uint32_t tme = me, top = op;
    if (n.sym) {
        // lexicographic order of the 9-tuple (cell 0 most significant) = order of sum t[i] * 3^(8-i)
        int best = 0x7fffffff;
        for (int s = 0; s < 8; s++) {
            int key = 0;
            for (int i = 0; i < 9; i++) {
                int c = syms[s][i];
                key = key * 3 + ((x >> c) & 1) + 2 * ((o >> c) & 1);
            }
            if (key < best) { best = key; pi = s; }
        }
        tme = 0; top = 0;
        for (int i = 0; i < 9; i++) {
            int c = syms[pi][i];
            tme |= ((me >> c) & 1u) << i;
            top |= ((op >> c) & 1u) << i;
        }
    }
    uint32_t in = tme | (top << 9);
    int acc[9];
    for (int k = 0; k < 9; k++) acc[k] = n.b2[k];
    for (int j = 0; j < n.H; j++) {
        int s;
        if (n.planes) s = POPC(n.pos[j] & in) - POPC(n.neg[j] & in);
        else s = POPC(n.pos[j] & tme) - POPC(n.pos[j] & top) - POPC(n.neg[j] & tme) + POPC(n.neg[j] & top);
        int z = s + n.b1[j];
        int h = n.step ? (z >= 0 ? 1 : 0) : (z > 0 ? z : 0);
        if (h)
            for (int k = 0; k < 9; k++) acc[k] += n.w2[k][j] * h;
    }
    uint32_t empty = ~(tme | top) & 0x1FF;
    int bm = -1, bz = 0;
    for (int k = 0; k < 9; k++)
        if (((empty >> k) & 1) && (bm < 0 || acc[k] > bz)) { bm = k; bz = acc[k]; }
    return n.sym ? syms[pi][bm] : bm;
}

// ---- opponents ------------------------------------------------------------------------------------------------
HD int opp_move(const Arm& a, const Tables& t, Rng& g, uint32_t x, uint32_t o) {
    uint32_t empty = ~(x | o) & 0x1FF;
    bool xmove = POPC(x) == POPC(o);
    uint32_t me = xmove ? x : o, op = xmove ? o : x;
    switch (a.opp) {
        case 0:
            return pick(t.opt[code3(x, o)], rng_next(g));
        case 2: {  // kid: take a win with prob p, else block with prob p, else random
            uint32_t w = completing(me, empty);
            if (w && rng_next(g) < a.thresh) return pick(w, rng_next(g));
            uint32_t b = completing(op, empty);
            if (b && rng_next(g) < a.thresh) return pick(b, rng_next(g));
            return pick(empty, rng_next(g));
        }
        case 3:  // mix: perfect with prob q, else random
            if (rng_next(g) < a.thresh) return pick(t.opt[code3(x, o)], rng_next(g));
            return pick(empty, rng_next(g));
        default:
            return pick(empty, rng_next(g));
    }
}

struct Result {
    int outcome;  // 0 X wins, 1 draw, 2 O wins
    int theory;   // minimax value of the position after the opening: 0 X wins, 1 draw, 2 O wins
    int plies;
    uint32_t seq;  // index of the move sequence among all sequences of its length, offset by length
};

HD Result play_game(const Net& n, const uint8_t (*syms)[9], const uint32_t* seq_off, const Tables& t, const Arm& a,
                    uint64_t game) {
    Rng g;
    rng_init(g, game, a);
    uint32_t x = 0, o = 0, seq = 0;
    int ply = 0;
    Result r;
    for (; ply < a.open_k; ply++) {
        uint32_t empty = ~(x | o) & 0x1FF;
        int m = pick(empty, rng_next(g));
        seq = seq * (9 - ply) + POPC(empty & ((1u << m) - 1));
        if (ply & 1) o |= 1u << m; else x |= 1u << m;
    }
    int v = t.valx[code3(x, o)];
    r.theory = v > 0 ? 0 : (v < 0 ? 2 : 1);
    r.outcome = 1;
    while (ply < 9) {
        int mover = (ply & 1) ? 2 : 1;
        int m = (a.side == 0 || a.side == mover) ? net_move(n, syms, x, o) : opp_move(a, t, g, x, o);
        uint32_t empty = ~(x | o) & 0x1FF;
        seq = seq * (9 - ply) + POPC(empty & ((1u << m) - 1));
        ply++;
        if (mover == 1) { x |= 1u << m; if (won(x)) { r.outcome = 0; break; } }
        else { o |= 1u << m; if (won(o)) { r.outcome = 2; break; } }
    }
    r.plies = ply;
    r.seq = seq_off[ply] + seq;
    return r;
}

// ---- host side ------------------------------------------------------------------------------------------------
struct HostData {
    Net net;
    uint8_t syms[8][9];
    uint32_t seq_off[10];
    std::vector<uint16_t> opt;
    std::vector<int8_t> valx;
};

static void die(const char* m) { fprintf(stderr, "ttt_gpu: %s\n", m); exit(2); }

static void read_exact(FILE* f, void* p, size_t n) {
    if (fread(p, 1, n, f) != n) die("short read");
}

// tables.bin: "TTT1", uint8 syms[8][9], uint16 opt[19683], int8 valx[19683]
// net.bin:    "TNET", int32 H, planes, step, sym, uint32 pos[H], neg[H], int32 b1[H], int32 w2[9][H], int32 b2[9]
static void load(HostData& d, const char* tables, const char* netp) {
    FILE* f = fopen(tables, "rb");
    if (!f) die("cannot open tables");
    char mg[4];
    read_exact(f, mg, 4);
    if (memcmp(mg, "TTT1", 4)) die("bad tables magic");
    read_exact(f, d.syms, sizeof d.syms);
    d.opt.resize(NCODES); d.valx.resize(NCODES);
    read_exact(f, d.opt.data(), NCODES * 2);
    read_exact(f, d.valx.data(), NCODES);
    fclose(f);
    f = fopen(netp, "rb");
    if (!f) die("cannot open net");
    read_exact(f, mg, 4);
    if (memcmp(mg, "TNET", 4)) die("bad net magic");
    int32_t hdr[4];
    read_exact(f, hdr, sizeof hdr);
    memset(&d.net, 0, sizeof d.net);
    d.net.H = hdr[0]; d.net.planes = hdr[1]; d.net.step = hdr[2]; d.net.sym = hdr[3];
    if (d.net.H < 1 || d.net.H > MAXH) die("H out of range");
    int H = d.net.H;
    read_exact(f, d.net.pos, 4 * H);
    read_exact(f, d.net.neg, 4 * H);
    read_exact(f, d.net.b1, 4 * H);
    for (int k = 0; k < 9; k++) read_exact(f, d.net.w2[k], 4 * H);
    read_exact(f, d.net.b2, 4 * 9);
    fclose(f);
    uint32_t acc = 0, perm = 1;
    for (int L = 0; L < 10; L++) {
        d.seq_off[L] = L >= 5 ? acc : 0;
        if (L >= 1) perm *= (10 - L);  // 9!/(9-L)!
        if (L >= 5) acc += perm;
    }
    if (acc != SEQ_TOTAL) die("seq offsets");
}

static Arm parse_arm(char** v) {
    Arm a;
    a.side = atoi(v[0]); a.opp = atoi(v[1]); a.thresh = (uint32_t)strtoul(v[2], 0, 10);
    a.open_k = atoi(v[3]); a.seed0 = (uint32_t)strtoul(v[4], 0, 10); a.seed1 = (uint32_t)strtoul(v[5], 0, 10);
    a.arm_id = (uint32_t)strtoul(v[6], 0, 10);
    if (a.side < 0 || a.side > 2 || a.opp < 0 || a.opp > 3 || a.open_k < 0 || a.open_k > 3) die("bad arm");
    if (a.side == 0 && a.open_k == 0) {}  // plain self-play is allowed (one game)
    return a;
}

#ifdef __CUDACC__
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "cuda: %s (%s:%d)\n", \
    cudaGetErrorString(e), __FILE__, __LINE__); exit(3); } } while (0)

__global__ void k_parity(const int* codes, int n, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = codes[i];
    uint32_t x = 0, o = 0;
    for (int k = 0; k < 9; k++) { int v = c % 3; c /= 3; if (v == 1) x |= 1u << k; else if (v == 2) o |= 1u << k; }
    out[i] = net_move(c_net, c_syms, x, o);
}

__global__ void k_trace(Tables t, Arm a, uint64_t start, int n, int4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Result r = play_game(c_net, c_syms, c_seq_off, t, a, start + i);
    out[i] = make_int4(r.outcome, r.theory, r.plies, (int)r.seq);
}

// counts[0..8] = theory * 3 + outcome, counts[9] = plies
__global__ void k_run(Tables t, Arm a, uint64_t start, uint64_t n, unsigned long long* counts, uint32_t* bitmap) {
    unsigned long long c[10] = {0};
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        Result r = play_game(c_net, c_syms, c_seq_off, t, a, start + i);
        c[r.theory * 3 + r.outcome]++;
        c[9] += r.plies;
        uint32_t w = r.seq >> 5, b = 1u << (r.seq & 31);
        if (!(bitmap[w] & b)) atomicOr(&bitmap[w], b);
    }
    for (int k = 0; k < 10; k++) {
        unsigned long long v = c[k];
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((threadIdx.x & 31) == 0 && v) atomicAdd(&counts[k], v);
    }
}

struct Dev {
    uint16_t* opt; int8_t* valx;
    Tables t;
};

static Dev upload(HostData& d) {
    Dev v;
    CK(cudaMemcpyToSymbol(c_net, &d.net, sizeof(Net)));
    CK(cudaMemcpyToSymbol(c_syms, d.syms, sizeof d.syms));
    CK(cudaMemcpyToSymbol(c_seq_off, d.seq_off, sizeof d.seq_off));
    CK(cudaMalloc(&v.opt, NCODES * 2));
    CK(cudaMalloc(&v.valx, NCODES));
    CK(cudaMemcpy(v.opt, d.opt.data(), NCODES * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v.valx, d.valx.data(), NCODES, cudaMemcpyHostToDevice));
    v.t.opt = v.opt; v.t.valx = v.valx;
    return v;
}
#endif

int main(int argc, char** argv) {
    if (argc < 5) die("usage: parity|trace|run tables.bin net.bin ...");
    HostData d;
    load(d, argv[2], argv[3]);
    Tables ht{d.opt.data(), d.valx.data()};
    const char* mode = argv[1];
    if (!strcmp(mode, "parity")) {
        FILE* f = fopen(argv[4], "r");
        if (!f) die("cannot open codes");
        std::vector<int> codes;
        int c;
        while (fscanf(f, "%d", &c) == 1) codes.push_back(c);
        fclose(f);
        int n = (int)codes.size();
        std::vector<int> out(n);
#ifdef __CUDACC__
        upload(d);
        int *dc, *dout;
        CK(cudaMalloc(&dc, 4 * n)); CK(cudaMalloc(&dout, 4 * n));
        CK(cudaMemcpy(dc, codes.data(), 4 * n, cudaMemcpyHostToDevice));
        k_parity<<<(n + 255) / 256, 256>>>(dc, n, dout);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dout, 4 * n, cudaMemcpyDeviceToHost));
#else
        for (int i = 0; i < n; i++) {
            int cc = codes[i];
            uint32_t x = 0, o = 0;
            for (int k = 0; k < 9; k++) { int v = cc % 3; cc /= 3; if (v == 1) x |= 1u << k; else if (v == 2) o |= 1u << k; }
            out[i] = net_move(d.net, d.syms, x, o);
        }
#endif
        for (int i = 0; i < n; i++) printf("%d\n", out[i]);
        return 0;
    }
    if (argc < 13) die("usage: trace|run tables net side opp thresh open_k seed0 seed1 arm_id start n [bitmap.out]");
    Arm a = parse_arm(argv + 4);
    uint64_t start = strtoull(argv[11], 0, 10), n = strtoull(argv[12], 0, 10);
    if (!strcmp(mode, "trace")) {
        if (n > (1u << 24)) die("trace n too large");
        std::vector<Result> res(n);
#ifdef __CUDACC__
        Dev v = upload(d);
        int4* dout;
        CK(cudaMalloc(&dout, sizeof(int4) * n));
        k_trace<<<(int)((n + 255) / 256), 256>>>(v.t, a, start, (int)n, dout);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<int4> h(n);
        CK(cudaMemcpy(h.data(), dout, sizeof(int4) * n, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < n; i++) res[i] = Result{h[i].x, h[i].y, h[i].z, (uint32_t)h[i].w};
#else
        for (uint64_t i = 0; i < n; i++) res[i] = play_game(d.net, d.syms, d.seq_off, ht, a, start + i);
#endif
        for (uint64_t i = 0; i < n; i++) printf("%d %d %d %u\n", res[i].outcome, res[i].theory, res[i].plies, res[i].seq);
        return 0;
    }
    if (!strcmp(mode, "run")) {
        if (argc < 14) die("run needs bitmap.out");
        unsigned long long counts[10] = {0};
        std::vector<uint32_t> bitmap(BITMAP_WORDS, 0);
        auto t0 = std::chrono::steady_clock::now();
        double kernel_ms = 0;
#ifdef __CUDACC__
        Dev v = upload(d);
        unsigned long long* dcounts;
        uint32_t* dbm;
        CK(cudaMalloc(&dcounts, sizeof counts)); CK(cudaMemset(dcounts, 0, sizeof counts));
        CK(cudaMalloc(&dbm, 4 * BITMAP_WORDS)); CK(cudaMemset(dbm, 0, 4 * BITMAP_WORDS));
        int sms = 0;
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        const char* bl = getenv("TTT_BLOCKS_PER_SM");
        int blocks = sms * (bl ? atoi(bl) : 16);
        const char* ln = getenv("TTT_LAUNCH_GAMES");
        uint64_t per = ln ? strtoull(ln, 0, 10) : (1ull << 30);  // games per launch (keeps each launch short)
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (uint64_t done = 0; done < n; done += per) {
            uint64_t m = n - done < per ? n - done : per;
            CK(cudaEventRecord(e0));
            k_run<<<blocks, 256>>>(v.t, a, start + done, m, dcounts, dbm);
            CK(cudaGetLastError());
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            kernel_ms += ms;
        }
        CK(cudaMemcpy(counts, dcounts, sizeof counts, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bitmap.data(), dbm, 4 * BITMAP_WORDS, cudaMemcpyDeviceToHost));
        const char* dev = "cuda";
#else
        for (uint64_t i = 0; i < n; i++) {
            Result r = play_game(d.net, d.syms, d.seq_off, ht, a, start + i);
            counts[r.theory * 3 + r.outcome]++;
            counts[9] += r.plies;
            bitmap[r.seq >> 5] |= 1u << (r.seq & 31);
        }
        const char* dev = "cpu";
#endif
        double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        FILE* f = fopen(argv[13], "wb");
        if (!f || fwrite(bitmap.data(), 4, BITMAP_WORDS, f) != BITMAP_WORDS) die("cannot write bitmap");
        fclose(f);
        printf("{\"device\":\"%s\",\"start\":%llu,\"n\":%llu,\"counts\":[", dev, (unsigned long long)start,
               (unsigned long long)n);
        for (int k = 0; k < 9; k++) printf("%s%llu", k ? "," : "", counts[k]);
        printf("],\"plies\":%llu,\"kernel_ms\":%.3f,\"wall_ms\":%.3f}\n", counts[9], kernel_ms, wall_ms);
        return 0;
    }
    die("unknown mode");
}
