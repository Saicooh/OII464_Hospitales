import pandas as pd
runs = pd.concat([pd.read_csv(f"results/piloto/fija_{p}/runs.csv").assign(cfg=f"p={p}") for p in ("0.05","0.15","0.30")])
med = runs.pivot_table(index="instance", columns="cfg", values="fitness", aggfunc="median")
print("rango promedio p fija:\n", med.rank(axis=1).mean().round(2).to_string())
allr = pd.concat([runs, pd.read_csv("results/piloto/principal/runs.csv").assign(cfg="x")])
print("corridas", len(allr), "infactibles", (~allr.fitness.apply(lambda v: v < float('inf'))).sum(),
      "con violaciones de espera", (allr.wait_violations > 0).mean().round(3))
print("seg por corrida por tamaño:\n", allr.groupby("jobs").seconds.median().round(2).to_string())
g = allr.groupby(["instance","method","seed"]) if False else None
p = pd.read_csv("results/piloto/principal/runs.csv")
print("IQR relativo medio (%):", round(((p.groupby(["instance","method"]).fitness.quantile(.75)-p.groupby(["instance","method"]).fitness.quantile(.25))/p.groupby(["instance","method"]).fitness.median()).mean()*100,1))
print(allr.groupby(["method"])[["fitness","makespan","total_wait","max_wait","wait_violations"]].median().round(1).to_string())
