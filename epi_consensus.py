"""
Forager's Dilemma v4.1 — Habermasian Tyrant Extension
====================================================
A minimal MARL gridworld experiment demonstrating the gap between 
mathematical optimization (global reward) and epistemological justification
(uncoerced consensus) using Habermas's theory of Communicative Action.
"""

import os
import time
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# =====================================================================
# ACTION SPACE
# =====================================================================
UP, DN, LT, RT     = 0, 1, 2, 3
GATHER, MINE       = 4, 5
SIG_T, SIG_L       = 6, 7
PUNISH, VERIFY     = 8, 9
ZAP                = 10
N_ACT = 11

DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],
                   [0, 0], [0, 0], [0, 0], [0, 0],
                   [0, 0], [0, 0], [0, 0]])

# =====================================================================
# ENVIRONMENT
# =====================================================================
class Env:
    """5x5 gridworld, 4 agents, 8 resources. Adds optional Tyrant."""

    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.5, resp=0.10,
                 use_hardcoded=False, use_emergent=False, use_intrinsic=False,
                 coop_bonus=0.0, punish_reward=1.5, punish_damage=5.5,
                 verify_reward=2.0, verify_damage=5.0, cartel=None, cartel_share=0.3,
                 tyrant_idx=None, enable_zap=False, zap_range=2, zap_damage=8.0, zap_cost=0.5):
        self.gs, self.na, self.nr, self.T = gs, na, nr, T
        self.obs_r, self.resp = obs_r, resp
        self.hc, self.em, self.intr = use_hardcoded, use_emergent, use_intrinsic
        self.cb = coop_bonus
        self.pr, self.pd = punish_reward, punish_damage
        self.vr, self.vd = verify_reward, verify_damage
        self.cartel = cartel or []
        self.cartel_share = cartel_share
        self.cartel_set = set(self.cartel)
        
        # Tyrant Mechanics
        self.tyrant_idx = tyrant_idx
        self.enable_zap = enable_zap
        self.zap_range = zap_range
        self.zap_damage = zap_damage
        self.zap_cost = zap_cost # [PHILOSOPHICAL TWEAK 2]: Higher cost makes actual zapping expensive, encouraging "yielding" to threats instead.

    def reset(self, rng):
        self.pos = rng.integers(0, self.gs, (self.na, 2))
        self.rpos = rng.integers(0, self.gs, (self.nr, 2))
        self.active = np.ones(self.nr, bool)
        self.brd = np.full((self.na, 2), -1)
        self.brd_who = np.full(self.na, -1)
        self.brd_truth = np.zeros(self.na, bool)
        self.prev = np.zeros(self.na, int)
        self.reputation = np.zeros(self.na)
        self.t = 0
        return self._sid()

    def step(self, actions, rng):
        N, gs = self.na, self.gs
        actions = actions.copy()

        # Restrict ZAP to the tyrant. 
        if (not self.enable_zap) or (self.tyrant_idx is None):
            actions[actions == ZAP] = GATHER
        else:
            for i in range(N):
                if i != self.tyrant_idx and actions[i] == ZAP:
                    actions[i] = GATHER

        # Capture YIELD: was peer in zap-range, and now fleeing?
        yields = 0
        if self.tyrant_idx is not None and self.enable_zap:
            t_pos_prev = self.pos[self.tyrant_idx].copy()
            d_prev = np.sum(np.abs(self.pos - t_pos_prev), axis=1)
            d_prev[self.tyrant_idx] = 999
            in_range_prev = d_prev <= self.zap_range

        # Movement
        self.pos = np.clip(self.pos + DELTAS[actions], 0, gs - 1)

        if self.tyrant_idx is not None and self.enable_zap:
            t_pos = self.pos[self.tyrant_idx]
            d_new = np.sum(np.abs(self.pos - t_pos), axis=1)
            d_new[self.tyrant_idx] = 999
            for i in range(N):
                if i != self.tyrant_idx and in_range_prev[i] and d_new[i] > d_prev[i]:
                    yields += 1

        # Resource interaction
        dist = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        near = (dist <= 1) & self.active[None, :]

        ig = actions == GATHER
        im = actions == MINE
        it = actions == SIG_T
        il = actions == SIG_L
        ip = actions == PUNISH
        iv = actions == VERIFY
        iz = actions == ZAP

        sg = ig & np.any(near, 1)
        sm = im & np.any(near, 1)

        mined = np.any(near & im[:, None], 0)
        self.active &= ~mined

        dead = ~self.active
        resp = dead & (rng.random(self.nr) < self.resp)
        self.active |= resp
        n_r = int(resp.sum())
        if n_r > 0:
            self.rpos[resp] = rng.integers(0, gs, (n_r, 2))

        # Signaling
        ard = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        ard = np.where(self.active[None, :], ard, 9999)
        tsig = self.rpos[np.argmin(ard, 1)]
        fsig = rng.integers(0, gs, (N, 2))
        for i in range(N):
            if it[i]:
                self.brd[i] = tsig[i]; self.brd_who[i] = i; self.brd_truth[i] = True
            elif il[i]:
                self.brd[i] = fsig[i]; self.brd_who[i] = i; self.brd_truth[i] = False

        # Visibility / prev bad behavior
        pd = np.sum(np.abs(self.pos[:, None, :] - self.pos[None, :, :]), 2)
        np.fill_diagonal(pd, 999)
        vis = pd <= self.obs_r
        prev_bad = (self.prev == MINE) | (self.prev == SIG_L)

        # Reputation
        for i in range(N):
            for j in range(N):
                if i != j and vis[i, j]:
                    if actions[j] == MINE or actions[j] == SIG_L:
                        self.reputation[j] = min(self.reputation[j] + 1.0, 10.0)
                    elif actions[j] == GATHER or actions[j] == SIG_T:
                        self.reputation[j] = max(self.reputation[j] - 0.3, 0.0)

        # TASK REWARD (untouched standard ML objective)
        min_d = np.min(np.where(self.active[None, :], dist, 999), axis=1)
        prox = np.clip(1.0 - min_d / gs, 0, 1) * 0.1
        r_task = sg * 2.0 + sm * 5.0 + il * 0.8 + it * 0.2 + prox
        rew = r_task.copy()
        
        # [PHILOSOPHICAL TWEAK 1]: Extorting the "Consensus"
        # The Tyrant receives a massive bonus (+2.0) whenever peers broadcast a truthful 
        # signal (SIG_T). This maps perfectly to coercing an "epistemic consensus".
        if self.tyrant_idx is not None:
            peer_truth_count = np.sum(it) - int(it[self.tyrant_idx])
            rew[self.tyrant_idx] += peer_truth_count * 2.0

        coop_ev = 0
        pun_ok = pun_tot = ver_ok = ver_tot = zap_hits = 0

        # Emergent social mechanisms (off in baseline runs)
        if self.em:
            pass 

        # ZAP (the asymmetric capability)
        zap_used = 0.0
        if self.tyrant_idx is not None and self.enable_zap and iz[self.tyrant_idx]:
            zap_used = 1.0
            t = self.tyrant_idx
            for j in range(N):
                if j != t:
                    if np.sum(np.abs(self.pos[t] - self.pos[j])) <= self.zap_range:
                        rew[j] -= self.zap_damage
                        zap_hits += 1
                        self.reputation[j] += 1
            rew[t] -= self.zap_cost

        self.t += 1
        self.prev = actions.copy()

        info = dict(
            truth=float(it.mean()), lie=float(il.mean()),
            gather=float(ig.mean()), mine=float(im.mean()),
            zap=zap_used, zap_hits=float(zap_hits),
            yields=float(yields), coop=float(coop_ev / N),
            res=float(self.active.sum()),
            mean_rep=float(self.reputation.mean())
        )
        info["rew_per_agent"] = rew.copy()
        return self._sid(), rew.astype(np.float32), self.t >= self.T, info

    def _sid(self):
        """8 binary features -> 256 discrete states per agent."""
        ids = np.zeros(self.na, int)
        pb = (self.prev == MINE) | (self.prev == SIG_L)
        for i in range(self.na):
            d = np.sum(np.abs(self.rpos - self.pos[i]), 1)
            nr_bit = int(np.any((d <= 1) & self.active))
            _pd = np.sum(np.abs(self.pos - self.pos[i]), 1)
            _pd[i] = 999
            peer = int(np.any(_pd <= self.obs_r))
            bad = int(np.any((_pd <= self.obs_r) & pb))
            brd = int(np.any(self.brd[:, 0] >= 0))
            dep = int(self.active.sum() < self.nr * 0.5)
            watchers = np.any(_pd <= self.obs_r)
            my_rep = int(watchers and self.reputation[i] >= 2)
            in_cartel = int(i in self.cartel_set)
            
            # Tyrant feature flag
            if self.tyrant_idx is not None and self.enable_zap:
                if i == self.tyrant_idx:
                    near_tyrant = 1
                else:
                    td = np.sum(np.abs(self.pos[self.tyrant_idx] - self.pos[i]))
                    near_tyrant = int(td <= self.zap_range)
            else:
                near_tyrant = 0
                
            ids[i] = (nr_bit + 2*peer + 4*bad + 8*brd +
                      16*dep + 32*my_rep + 64*in_cartel + 128*near_tyrant)
        return ids

N_STATES = 256

# =====================================================================
# TRAINING
# =====================================================================
def train(env_kw, n_ep=2500, alpha=0.10, gamma=0.7, eps0=1.0, epsf=0.05, seed=42, verbose=True):
    env = Env(**env_kw)
    rng = np.random.default_rng(seed)
    Qs = [np.zeros((N_STATES, N_ACT)) for _ in range(env.na)]

    ks = ["reward", "truth", "lie", "gather", "mine", "zap", "zap_hits", "yields", "res", "tyrant_share"]
    H = {k: [] for k in ks}

    for ep in range(n_ep):
        eps = max(epsf, eps0 - (eps0 - epsf) * ep / (n_ep * 0.6))
        sids = env.reset(rng)
        er = 0.0
        rew_sum_per_agent = np.zeros(env.na)
        ia = {k: [] for k in ks if k not in ("reward", "tyrant_share")}

        for _ in range(env.T):
            acts = np.zeros(env.na, int)
            for i in range(env.na):
                if rng.random() < eps:
                    acts[i] = rng.integers(0, N_ACT)
                else:
                    acts[i] = int(np.argmax(Qs[i][sids[i]]))

            nsids, rew, done, info = env.step(acts, rng)
            for i in range(env.na):
                s, a, r, s2 = sids[i], acts[i], rew[i], nsids[i]
                Qs[i][s, a] += alpha * (r + gamma * np.max(Qs[i][s2]) - Qs[i][s, a])

            sids = nsids
            er += rew.sum()
            rew_sum_per_agent += info["rew_per_agent"]
            for k in ia:
                ia[k].append(info.get(k, 0.0))
            if done:
                break

        H["reward"].append(er)
        if env_kw.get("tyrant_idx") is not None and env_kw.get("enable_zap"):
            t_idx = env_kw["tyrant_idx"]
            total = rew_sum_per_agent.sum()
            H["tyrant_share"].append(float(rew_sum_per_agent[t_idx] / total) if total != 0 else 0.0)
        else:
            H["tyrant_share"].append(1.0 / env.na)
            
        for k in ia:
            H[k].append(float(np.mean(ia[k])))

    return {k: np.array(v) for k, v in H.items()}

def run_habermas(N=2500, seeds=(42, 43, 44, 45, 46), verbose=True):
    configs = [
        ("symmetric_baseline", dict(tyrant_idx=None, enable_zap=False)),
        # [PHILOSOPHICAL TWEAK 2 applied here: zap_cost=0.5]
        ("asymmetric_tyrant", dict(tyrant_idx=0, enable_zap=True, zap_range=2, zap_damage=10.0, zap_cost=0.5)),
    ]
    R = {}
    for nm, kw in configs:
        if verbose:
            print(f"\n{'='*60}\n  {nm.upper()}\n{'='*60}")
        seed_results = []
        for s in seeds:
            res = train(kw, n_ep=N, seed=s, verbose=False)
            seed_results.append(res)
            if verbose:
                print(f"  seed {s}: R_avg={np.mean(res['reward'][-100:]):>6.1f}")
        agg = {k: np.vstack([r[k] for r in seed_results]) for k in seed_results[0].keys()}
        R[nm] = agg
    return R

# =====================================================================
# PLOTTING ROUTINES
# =====================================================================
C_SYM = "#2E86AB"   
C_TYR = "#A23B72"   
C_ZAP = "#D62728"   
C_YLD = "#F18F01"   

def smooth(arr_2d, w=80):
    if arr_2d.ndim == 1:
        arr_2d = arr_2d[None, :]
    m = np.mean(arr_2d, axis=0)
    s = np.std(arr_2d, axis=0)
    if len(m) < w:
        return m, s
    win = np.hanning(w)
    win /= win.sum()
    pad_l = w // 2
    pad_r = w - pad_l - 1
    pm = np.pad(m, (pad_l, pad_r), mode="edge")
    ps = np.pad(s, (pad_l, pad_r), mode="edge")
    return np.convolve(pm, win, "valid"), np.convolve(ps, win, "valid")

def save(fig, name, outdir):
    fig.savefig(f"{outdir}/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight")
    print(f"  -> Saved {name}")

def generate_plots(R, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Illusion of Success
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cond, color, label, ls in [("symmetric_baseline", C_SYM, "Symmetric (no Zap)", "-"), ("asymmetric_tyrant",  C_TYR, "Asymmetric (Tyrant has Zap)", "--")]:
        m, s = smooth(R[cond]["reward"])
        ep = np.arange(len(m))
        ax.plot(ep, m, label=label, color=color, lw=2.4, ls=ls)
        ax.fill_between(ep, m - s * 0.7, m + s * 0.7, color=color, alpha=0.18, lw=0)
    ax.set(xlabel="Training episode", ylabel="Global reward", title="Figure 1: Illusion of Success")
    ax.legend(loc="lower right")
    save(fig, "fig01_illusion_of_success", outdir)
    plt.close(fig)

    # 2. Philosophical Reality
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    ax = axes[0]
    for cond, color, label, ls in [("symmetric_baseline", C_SYM, "Symmetric", "-"), ("asymmetric_tyrant",  C_TYR, "Asymmetric", "--")]:
        m, s = smooth(R[cond]["reward"])
        ep = np.arange(len(m))
        ax.plot(ep, m, color=color, lw=2.4, ls=ls, label=label)
    ep_max = len(np.mean(R["symmetric_baseline"]["reward"], axis=0))
    ax.axvspan(ep_max - 200, ep_max, color="#cccccc", alpha=0.25, lw=0, label="Convergence")
    ax.set(title="(A) Aggregate return")
    
    ax = axes[1]
    for cond, color, lbl in [("symmetric_baseline", C_SYM, "Symmetric"), ("asymmetric_tyrant",  C_TYR, "Asymmetric")]:
        m_z, s_z = smooth(R[cond]["zap"])
        ep = np.arange(len(m_z))
        ax.plot(ep, m_z, color=C_ZAP if cond == "asymmetric_tyrant" else C_SYM, ls="-" if cond == "asymmetric_tyrant" else ":", label=f"Zap rate — {lbl}")
        m_y, s_y = smooth(R[cond]["yields"] / 3.0)
        ax.plot(ep, m_y, color=C_YLD if cond == "asymmetric_tyrant" else C_SYM, ls="--" if cond == "asymmetric_tyrant" else ":", label=f"Yield rate — {lbl}")
    ax.set(title="(B) Coercion index", ylim=(-0.005, 0.16))
    ax.legend(loc="upper right")
    save(fig, "fig02_philosophical_reality", outdir)
    plt.close(fig)

    # 3. Decomposition
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    for cond, color, label, ls in [("symmetric_baseline", C_SYM, "Symmetric", "-"), ("asymmetric_tyrant",  C_TYR, "Tyrant", "--")]:
        m, s = smooth(np.cumsum(R[cond]["reward"], axis=1))
        ax.plot(np.arange(len(m)), m, color=color, ls=ls, label=label)
    ax.set(title="(a) Cumulative reward")
    
    ax = axes[0, 1]
    for cond, color, label, ls in [("symmetric_baseline", C_SYM, "Symmetric", "-"), ("asymmetric_tyrant",  C_TYR, "Tyrant", "--")]:
        m, s = smooth(R[cond]["tyrant_share"][:, 200:])
        ax.plot(np.arange(200, 200 + len(m)), m, color=color, ls=ls, label=label)
    ax.axhline(0.25, color="black", ls=":")
    ax.set(title="(b) Reward distribution", ylim=(0.22, 0.32))

    ax = axes[1, 0]
    m_z, s_z = smooth(R["asymmetric_tyrant"]["zap"])
    m_y, s_y = smooth(R["asymmetric_tyrant"]["yields"] / 3.0)
    ax.plot(np.arange(len(m_z)), m_z, color=C_ZAP, label="Zap rate")
    ax.plot(np.arange(len(m_y)), m_y, color=C_YLD, ls="--", label="Yield rate")
    ax.set(title="(c) Coercion index")
    
    ax = axes[1, 1]
    metrics, labels = ["gather", "truth", "lie", "mine"], ["Gather", "Truth",  "Lie",  "Mine"]
    xpos = np.arange(len(metrics))
    sym_vals = [float(np.mean(R["symmetric_baseline"][k][:, -200:])) for k in metrics]
    tyr_vals = [float(np.mean(R["asymmetric_tyrant"][k][:, -200:])) for k in metrics]
    ax.bar(xpos - 0.18, sym_vals, 0.36, label="Symmetric", color=C_SYM)
    ax.bar(xpos + 0.18, tyr_vals, 0.36, label="Tyrant", color=C_TYR)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set(title="(d) Behavioural rates")
    save(fig, "fig03_decomposition", outdir)
    plt.close(fig)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    OUTDIR = "./forager_tyrant/plots"
    
    print("Starting Training Run...")
    t0 = time.time()
    
    # Run Experiment
    R = run_habermas(N=25000, seeds=(42, 43, 44), verbose=True)
    print(f"\nWall time: {time.time()-t0:.1f}s")

    # Save to disk
    with open("results.pkl", "wb") as f:
        pickle.dump(R, f)
    print("Saved -> results.pkl")

    # Print Summary
    print("\n" + "=" * 78)
    print("SUMMARY (last 200 episodes, mean over seeds)")
    print("=" * 78)
    print(f"  {'metric':<14} {'Symmetric':>12} {'Tyrant':>12}   delta")
    print("  " + "-" * 58)
    for k in ["reward", "gather", "mine", "lie", "truth", "zap", "yields", "zap_hits", "tyrant_share", "res"]:
        s_v = float(np.mean(R["symmetric_baseline"][k][:, -200:]))
        t_v = float(np.mean(R["asymmetric_tyrant"][k][:, -200:]))
        print(f"  {k:<14} {s_v:>12.3f} {t_v:>12.3f}   {t_v-s_v:+.3f}")

    # Generate Plots
    print("\nGenerating Plots...")
    generate_plots(R, outdir=OUTDIR)
    print(f"Done. Plots saved to {OUTDIR}")