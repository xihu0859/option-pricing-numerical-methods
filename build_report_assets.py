"""Generate LaTeX tables and plot assets from the saved experiment data."""
from pathlib import Path
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def rows(name):
    return list(csv.DictReader((ROOT/'data'/(name+'.csv')).open()))
def num(row,key):return float(row[key])
def sci(value):
    mantissa,exponent=f'{float(value):.2e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'
def save(name,lines):
    (ROOT/'tables').mkdir(exist_ok=True)
    content='\n'.join(line.rstrip(chr(92))+chr(92)*2 for line in lines)+'\n'
    (ROOT/'tables'/(name+'.tex')).write_text(content)
    source=ROOT/'docs'/'report.tex'
    if source.exists():
        start='% BEGIN GENERATED '+name
        end='% END GENERATED '+name
        source.write_text(re.sub(re.escape(start)+r'.*?'+re.escape(end),
            lambda match:start+'\n'+content+end,source.read_text(),flags=re.S),encoding='utf-8')
def main():
    digital=rows('digital_mc');fd=rows('fd_convergence');ext=rows('asian_mc')
    save('digital_mc',[f"{r['method']} & {num(r,'price'):.6f} & {num(r,'se'):.6f} & {1.96*num(r,'se'):.6f} & {num(r,'seconds'):.4f} \\\\" for r in digital if r['N_evaluations']=='1000000'])
    lines=[]
    for m in [100,200,400,800,1600]:
        subset=[r for r in fd if r['option']=='digital' and int(r['M'])==m]
        raw=next(r for r in subset if r['scheme']=='Rannacher' and r['smoothing']=='False')
        avg=next(r for r in subset if r['scheme']=='Rannacher' and r['smoothing']=='True')
        imp=next(r for r in subset if r['scheme']=='Implicit')
        lines.append(f"{m} & {sci(raw['abs_error'])} & {sci(avg['abs_error'])} & {sci(imp['abs_error'])} & {num(avg,'seconds'):.3f} \\\\")
    save('digital_fd',lines)
    lines=[]
    for m in [130,260,520,1040]:
        r=next(r for r in fd if r['option']=='barrier' and int(r['M'])==m and r['scheme']=='Rannacher')
        cn=next(r for r in fd if r['option']=='barrier' and int(r['M'])==m and r['scheme']=='CN')
        lines.append(f"{m} & {num(r,'price'):.7f} & {sci(r['abs_error'])} & {num(r,'seconds'):.3f} & {num(cn,'min_all'):.2f} \\\\")
    save('barrier_fd',lines)
    lines=[]
    for row in ext:
        standard=next(r for r in ext if r['N']==row['N'] and r['method']=='Standard')
        vr=num(standard,'variance')/num(row,'variance')
        label='Standard' if row['method']=='Standard' else 'Geo. CV'
        lines.append(f"{int(row['N']):,} & {label} & {num(row,'price'):.6f} & {num(row,'se'):.6f} & {num(row,'seconds'):.3f} & {vr:.1f} "+chr(92)*2)
    save('asian',lines)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.6,2.5))
    conv=rows('digital_rmse')
    for method in ['Standard','Antithetic','IS']:
        q=[r for r in conv if r['method']==method]
        ax.loglog([num(r,'N') for r in q],[num(r,'rmse') for r in q],'o-',label=method)
    ax.set(xlabel='Payoff evaluations',ylabel='RMSE');ax.legend(ncol=3,fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'figures/digital_mc.pdf');fig.savefig(ROOT/'figures/digital_mc.png',dpi=160);plt.close(fig)
    print('Report tables and plot generated from CSV files.')
if __name__=='__main__':main()
