import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="serif"; plt.rcParams["font.serif"]=["Liberation Serif","Times New Roman","DejaVu Serif"]
fig,ax=plt.subplots(figsize=(10.5,3.6),dpi=200); ax.set_xlim(-1.5,101.5); ax.set_ylim(0,36); ax.axis("off")
steps=[("ISCXVPN2016\n16 task ARFF files\n(4 timeouts)","#e8eef7"),
("Column audit\nvalue identities\n(100% of rows)","#e8eef7"),
("Cleaning\nexact duplicates\nremoved per file","#e8eef7"),
("Stratified split\n70 / 15 / 15\n(seed 42)","#fdf1dc"),
("Model search\n7 method types\n5-fold CV, 50 trials","#fdf1dc"),
("Selection on\nvalidation\n(macro F1)","#fdf1dc"),
("Test set\nscored once\n+ bootstrap, Holm","#e3f2e3")]
w,h,y=12.4,14,19; gap=(100-7*w)/6
xs=[i*(w+gap) for i in range(7)]
for x,(t,c) in zip(xs,steps):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=c,ec="#333",lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.6)
for x in xs[:-1]:
    ax.annotate("",xy=(x+w+gap-0.3,y+h/2),xytext=(x+w+0.4,y+h/2),arrowprops=dict(arrowstyle="-|>",lw=1.1,color="#333"))
# bottom row: post-test analyses
bt=[("Isotonic calibration\n(ECE, 15 bins)",),("SHAP + feature-group\nablation",),("Stability, learning\ncurve, Isnan control",),("CLI predictor and\nStreamlit demo",)]
bw=20; bgap=(100-4*bw)/3
for i,(t,) in enumerate(bt):
    x=i*(bw+bgap)
    ax.add_patch(FancyBboxPatch((x,1.5),bw,9,boxstyle="round,pad=0.3,rounding_size=1.2",fc="#f3f3f3",ec="#777",lw=0.9,ls="--"))
    ax.text(x+bw/2,6,t,ha="center",va="center",fontsize=8.6)
ax.text(50,14.6,"Supporting analyses of the final models",ha="center",fontsize=8,style="italic",color="#444")
ax.text(xs[0],34.8,"Data stage",fontsize=8.5,weight="bold",color="#2a4d7f")
ax.text(xs[3],34.8,"Leakage-safe modelling (training + validation rows only)",fontsize=8.5,weight="bold",color="#8a5a00")
ax.text(xs[6]-2,34.8,"Held-out evaluation",fontsize=8.5,weight="bold",color="#2d6a2d")
plt.savefig("fig_pipeline.png",bbox_inches="tight",dpi=220)
