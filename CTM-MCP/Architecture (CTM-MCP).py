"""
CTM + MCP CONTROL STACK  (final architecture, October 2025)

Blocks:
  • Multi-Modal Sensor Ingestion
  • Perception Encoder   (VisionBackbone, AudioSpectroCNN, JointAngleMLP, FusionDense)
  • Tick-Slab CTM Runtime
        SynapseOperator · TickSlabFusionUnit · LowRank μ-MLP Bank
        SyncRank1Updater · CertaintyEstimator
  • ★ ConsensusAggregator  (new)
  • Emotional State Loop   (AffectDec, CertMod)
  • MCP Envelope Router    (Serializer*, PolicyGate, ParallelBranchMgr, TimeoutSafePass)
  • Actuator Controller    (TrajectoryPlanner, ComplianceFilter, TorqueMapper)
"""

import random, math, copy, time, json
random.seed(456)

# ───────────────────────── helpers ──────────────────────────
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def softmax(v): m = max(v); ex = [math.exp(x - m) for x in v]; s = sum(ex); return [e / s for e in ex]
def entropy(p): return -sum(pi * math.log(pi + 1e-12) for pi in p)
def l2(v): return math.sqrt(sum(x * x for x in v))
def dense_tanh(x, W): return [math.tanh(dot(w, x)) for w in W]
def make_dense(r, c, s=0.2): return [[random.uniform(-s, s) for _ in range(c)] for _ in range(r)]

# ───────────────────────── sensors ──────────────────────────
class DemoLoader:
    def __iter__(self):
        for _ in range(50):
            yield ([random.random() for _ in range(4096)],
                   [random.random() for _ in range(256)],
                   [random.uniform(-1, 1) for _ in range(12)],
                   [random.random() for _ in range(64)])

# ────────────────── perception encoder blocks ───────────────
class VisionBackbone:
    def __init__(self): self.W = make_dense(128, 4096)
    def forward(self, x): return dense_tanh(x, self.W)

class AudioSpectroCNN:
    def __init__(self): self.W = make_dense(64, 256)
    def forward(self, x): return dense_tanh(x, self.W)

class JointAngleMLP:
    def __init__(self): self.W = make_dense(32, 76)
    def forward(self, j, t): return dense_tanh(j + t, self.W)

class FusionDense:
    def __init__(self): self.W = make_dense(256, 224)
    def forward(self, v): return dense_tanh(v, self.W)

# ────────────────── tick-slab CTM components ────────────────
LOGIT_SCALE = 8.0
DECAY       = 0.999
BETA        = 0.9          # gated carry retention

class SynapseOperator:
    def __init__(self, D): self.W = make_dense(D, D + 256)
    def fwd(self, z, f): return dense_tanh(z + f, self.W)

class LowRankBank:
    def __init__(self, D, M, r=8):
        self.D, self.M, self.r = D, M, r
        self.A = make_dense(M, r, 0.2)
        self.B = make_dense(D, r, 0.2)
        self.b0 = [random.uniform(-.05, .05) for _ in range(D)]
    def tick(self, H):
        out = []
        for d in range(self.D):
            Ab = [sum(self.A[m][j] * self.B[d][j] for j in range(self.r))
                  for m in range(self.M)]
            pre = sum(Ab[m] * H[d][m] for m in range(self.M)) + self.b0[d]
            out.append(math.tanh(pre))
        return out

class SyncRank1Upd:
    def __init__(self, D, P, decay=DECAY):
        self.decay = decay
        self.pairs = [(i, (7 * i) % D) for i in range(P)]
    def update(self, S, Z):
        L = len(Z)
        S = [v * (self.decay ** L) for v in S]
        for j, z in enumerate(Z):
            w = self.decay ** (L - 1 - j)
            for idx, (p, q) in enumerate(self.pairs):
                S[idx] += w * z[p] * z[q]
        return S

class CertHead:
    def __init__(self, P, C=4):
        self.W = make_dense(C, P, LOGIT_SCALE)
        self.C = C
    def cert(self, S):
        return 1 - entropy(softmax([dot(w, S) for w in self.W])) / math.log(self.C)
    def logits(self, S):  # helpful for aggregator
        return [dot(w, S) for w in self.W]

# ────────────────── affect + certainty loop ─────────────────
class AffectDec:
    def __init__(self, P):
        self.h1 = make_dense(32, P)
        self.h2 = make_dense(8, 32)
    def fwd(self, S):
        return dense_tanh(dense_tanh(S, self.h1), self.h2)

class CertMod:
    def __init__(self, eps0=0.75, alpha=0.5, delta=0.05):
        self.eps0, self.alpha, self.delta = eps0, alpha, delta
        self.last = 0.0
    def eps(self, e):
        mag = l2(e)
        if abs(mag - self.last) > self.delta:
            self.last = mag
        return self.eps0 * (1 + self.alpha * mag)

# ────────────────── MCP routing helpers ────────────────────
class PolicyGate:
    def __init__(self, gamma): self.gamma = gamma
    def route(self, c): return "planner" if c < self.gamma else "actr"

class Serializer:
    def pack(self, sync, slabs, affect):
        return json.dumps(dict(sync256=sync, slabs=slabs[-8:], affect=affect))

class TimeoutPass:
    def __init__(self, t=0.3):
        self.t=t; self.sync=None; self.stamp=time.time()
    def update(self, s): self.sync, self.stamp = s, time.time()
    def poll(self): return self.sync if self.sync and time.time()-self.stamp > self.t else None

class BranchMgr:
    """Spawn N CTM clones; return first that halts, otherwise last clone."""
    def __init__(self, n): self.n=n
    def race(self, ctor):
        last=None
        for _ in range(self.n):
            res = ctor()
            if res["halt"]: return res
            last = res
        return last  # guarantees sync vector exists

# ────────────────── ConsensusAggregator ★ ──────────────────
class ConsensusAggregator:
    def __init__(self, C=4): self.C = C
    def merge(self, sync_list, logit_list, fallback):
        """Return confidence-weighted average; if lists empty, return fallback."""
        if not sync_list:
            return fallback
        certs=[]
        for h in logit_list:
            p = softmax(h)
            c = 1 - entropy(p) / math.log(self.C)
            certs.append(c)
        w_sum = sum(certs) + 1e-9
        merged = [0.0] * len(sync_list[0])
        for s, c in zip(syncs:=sync_list, certs):
            for i, v in enumerate(s):
                merged[i] += c * v
        return [v / w_sum for v in merged]

# ────────────────── actuator stubs ─────────────────────────
class TrajPlanner:
    def comp(self, sync): return [sum(sync[i::12]) * 0.02 for i in range(12)]
class ComplFilter:
    def filt(self, tau): return [x * 0.8 for x in tau]
class TorqueMapper:
    def cur(self, tau): return [int(x * 1000) for x in tau]

# ────────────────── CTM wrapper ────────────────────────────
class CTM:
    def __init__(self, D=128, P=64, k=6, M=8):
        self.D, self.P, self.k, self.M = D, P, k, M
        self.z   = [0.0] * D
        self.H   = [[0.0] * M for _ in range(D)]
        self.S   = [0.0] * P
        self.hist=[]
        self.syn  = SynapseOperator(D)
        self.bank = LowRankBank(D, M)
        self.upd  = SyncRank1Upd(D, P)
        self.head = CertHead(P)
    def copy(self): return copy.deepcopy(self)

    def run(self, f_vec, eps, max_slabs=3):
        slabs=0; sync=None; logits=None
        while slabs<max_slabs:
            slabs+=1; Z=[]
            for _ in range(self.k):
                syn=self.syn.fwd(self.z, f_vec)
                for d,v in enumerate(syn): self.H[d]=[v]+self.H[d][:self.M-1]
                self.z=self.bank.tick(self.H); Z.append(self.z)
            self.S=self.upd.update(self.S, Z)
            c=self.head.cert(self.S)
            self.hist.append(self.S.copy())
            if c>=eps:
                sync=self.S.copy(); logits=self.head.logits(self.S)
                return dict(halt=True,c=c,sync=sync,logits=logits,slabs=slabs)
        # non-halting case still returns sync/logits
        return dict(halt=False,c=c,sync=self.S.copy(),
                    logits=self.head.logits(self.S), slabs=slabs)

# ────────────────── experiment orchestrator ─────────────────
class Experiment:
    def __init__(self, branches=3, gamma=0.45):
        # perception
        self.V=VisionBackbone(); self.A=AudioSpectroCNN()
        self.P=JointAngleMLP();  self.F=FusionDense()
        # loops
        self.aff=AffectDec(64); self.mod=CertMod()
        self.ser=Serializer();  self.pg=PolicyGate(gamma)
        self.tsp=TimeoutPass(); self.br=BranchMgr(branches)
        self.aggr=ConsensusAggregator()
        self.ctm=CTM()
        # actuator
        self.tr=TrajPlanner(); self.cf=ComplFilter(); self.tq=TorqueMapper()
        # metrics
        self.frames=self.plan=self.act=0; self.c_sum=self.s_sum=0

    def perception(self, frame):
        v,a,j,t=frame
        return self.F.forward(self.V.forward(v)+self.A.forward(a)+self.P.forward(j,t))

    def step(self, frame):
        f_vec=self.perception(frame)
        eps=self.mod.eps(self.aff.fwd(self.ctm.S))
        # spawn & race
        def make(): return self.ctm.copy().run(f_vec, eps)
        res=self.br.race(make)
        syncs=[res["sync"]] if res["sync"] else []
        logits=[res["logits"]] if res["logits"] else []
        # wait one extra slab for losers
        loser=self.ctm.copy().run(f_vec, eps)
        if loser["sync"]: syncs.append(loser["sync"]); logits.append(loser["logits"])
        merged=self.aggr.merge(syncs, logits, fallback=self.ctm.S)
        # MCP routing
        self.tsp.update(merged)
        dest=self.pg.route(res["c"])
        if self.tsp.poll(): dest="actr"
        if dest=="actr":
            self.tq.cur(self.cf.filt(self.tr.comp(merged))); self.act+=1
        else:
            self.ctm.z=[BETA*z_o+(1-BETA)*z_n for z_o,z_n in zip(self.ctm.z, merged)]
            self.plan+=1
        self.frames+=1; self.c_sum+=res["c"]; self.s_sum+=res["slabs"]

    def run(self):
        for f in DemoLoader(): self.step(f)
        self.dump()
    def dump(self):
        print("\n=== Experiment summary ===")
        print(dict(frames=self.frames,planner=self.plan,actuator=self.act,
                   mean_cert=round(self.c_sum/self.frames,4),
                   mean_slabs=round(self.s_sum/self.frames,2)))

# ─────────────────────────── entry ──────────────────────────
if __name__=="__main__":
    Experiment(branches=3, gamma=0.45).run()
