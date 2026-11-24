import pandas as pd, numpy as np
L = {"base":"A1","fija":"A2","aleatoria":"A3","reglas":"A4","aprende_p":"A5","aprende_mov":"A6","dos_niveles":"A7"}
r = pd.read_csv("results/final/runs.csv"); r["A"]=r.method.map(L)
a1 = r[r.A=="A1"]
comp = pd.DataFrame({"Cmax":a1.makespan/a1.fitness,"betaW":0.5*a1.total_wait/a1.fitness,"gammaMax":1.4*a1.max_wait/a1.fitness})
comp["resto"]=1-comp.sum(axis=1)
print("A1 composición mediana de f (%):", (comp.median()*100).round(1).to_dict())
print("max_wait mediana por variante:", r.groupby("A").max_wait.median().round(1).to_dict())
print("violaciones: % corridas >0:", r.groupby("A").wait_violations.apply(lambda v:(v>0).mean()*100).round(1).to_dict())
print("violaciones media trabajos:", r.groupby("A").wait_violations.mean().round(2).to_dict())
print("seconds mediana por jobs:", r.groupby("jobs").seconds.median().round(2).to_dict())
b = pd.read_csv("results/final/blocks.csv"); b["A"]=b.method.map(L)
a3=b[b.A=="A3"]
print("A3 recompensa media por p (x1e-3):", (a3.groupby("p").reward.mean()*1e3).round(2).to_dict())
print("A3 % bloques con mejora por p:", a3.groupby("p").reward.apply(lambda x:(x>1e-6).mean()*100).round(1).to_dict())
print("% bloques sin mejora:", b.groupby("A").reward.apply(lambda x:(x<=1e-6).mean()*100).round(1).to_dict())
for name in ("A5","A7"):
    s=b[b.A==name]
    g=s.groupby(["instance","seed"]).p
    dom=g.agg(lambda x:x.value_counts(normalize=True).iloc[0])
    which=g.agg(lambda x:x.value_counts().index[0])
    print(name,"uso p dominante medio %:",round(dom.mean()*100,1),"cuál domina %:",(which.value_counts(normalize=True)*100).round(1).to_dict())
for name in ("A6","A7"):
    s=r[r.A==name]
    print(name,"V final:",s[["q_intercambio","q_insercion","q_sala"]].mean().round(3).to_dict())
for name in ("A1","A2","A6","A7"):
    s=r[r.A==name]; c=s[["n_intercambio","n_insercion","n_sala"]].sum()
    if c.sum(): print(name,"uso mov %:",(c/c.sum()*100).round(1).to_dict())
