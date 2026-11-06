"""Reproduce the assessment experiments: python reproduce.py.
Model-generated data, not historical market observations. Fixed independent seeds.
All prices and SEs use discounted payoffs. Timing excludes plotting and file I/O.
"""
from pathlib import Path
import csv, json, time, platform
import numpy as np
import scipy
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; FIG=ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
S0=100.; R=.05; SIG=.25; T=1.; MU=R-SIG**2/2; DISC=np.exp(-R*T)
BASE=20260914

def save(name,rows):
    with (DATA/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def stats(y):
    v=float(np.var(y,ddof=1)); se=np.sqrt(v/len(y)); p=float(np.mean(y))
    return dict(price=p,se=float(se),ci_low=p-1.96*se,ci_high=p+1.96*se,variance=v)

def digital_exact(k=140.):
    a=(np.log(k/S0)-MU*T)/(SIG*np.sqrt(T))
    return float(DISC*ndtr(-a))

def bridge_weight(x,y,dt,h=np.log(130.)):
    ok=(x<h)&(y<h)
    return np.where(ok,-np.expm1(-2*np.maximum(h-x,0)*np.maximum(h-y,0)/(SIG**2*dt)),0.)

def barrier_ref():
    # Killed transition density from reflection principle; separate bridge check.
    x=np.log(S0); h=np.log(130.); sd=SIG*np.sqrt(T)
    def normal(y,m): return np.exp(-.5*((y-m)/sd)**2)/(sd*np.sqrt(2*np.pi))
    def killed(y):
        return normal(y,x+MU*T)-np.exp(2*MU*(h-x)/SIG**2)*normal(y,2*h-x+MU*T)
    p,e=quad(lambda y: DISC*(np.exp(y)-100)*killed(y),np.log(100),h,epsabs=1e-12,epsrel=1e-12)
    b,be=quad(lambda y: DISC*(np.exp(y)-100)*normal(y,x+MU*T)*bridge_weight(x,y,T),np.log(100),h,epsabs=1e-12,epsrel=1e-12)
    assert abs(p-b)<1e-11
    return p,e,b

def digital_sample(n,method,seed,k=140.):
    rng=np.random.default_rng(seed)
    a=(np.log(k/S0)-MU*T)/(SIG*np.sqrt(T))
    if method=='Antithetic':
        z=rng.standard_normal(n//2)
        return DISC*((z>a).astype(float)+(-z>a).astype(float))/2
    z=rng.standard_normal(n)
    if method=='IS': return DISC*((z+a)>a)*np.exp(-a*z-a*a/2)
    return DISC*(z>a)

def fd(kind,M,N,scheme='CN',smooth=True,smax=None,snapshot=False,strike=None):
    # Central differences in S; n=N is maturity-to-valuation time.
    top=(400. if kind=='digital' else 130.) if smax is None else smax
    s=np.linspace(0,top,M+1); ds=top/M; j=np.arange(1,M,dtype=float)
    a=.5*(SIG**2*j*j-R*j); b=-SIG**2*j*j-R; c=.5*(SIG**2*j*j+R*j)
    k=(140 if kind=='digital' else 100) if strike is None else strike
    u=(s>k).astype(float) if kind=='digital' else np.maximum(s-k,0)
    if kind=='digital' and smooth:
        u=np.clip((s+ds/2-k)/ds,0,1) # exact cell average of the step payoff
    u[0]=0; u[-1]=1 if kind=='digital' else 0
    dt=T/N; first=None; minall=float(u.min()); maxall=float(u.max())
    def boundary(tau): return np.exp(-R*tau) if kind=='digital' else 0.
    def advance(v,tau,h,theta):
        rhs=v[1:-1]+(1-theta)*h*(a*v[:-2]+b*v[1:-1]+c*v[2:])
        rhs[-1]+=theta*h*c[-1]*boundary(tau+h)
        if theta:
            ab=np.zeros((3,M-1)); ab[0,1:]=-theta*h*c[:-1]
            ab[1]=1-theta*h*b; ab[2,:-1]=-theta*h*a[1:]
            inside=solve_banded((1,1),ab,rhs,check_finite=False)
        else: inside=rhs
        return np.r_[0.,inside,boundary(tau+h)]
    for n in range(N):
        if scheme=='Rannacher' and n==0:
            u=advance(u,0,dt/2,1); u=advance(u,dt/2,dt/2,1)
        else: u=advance(u,n*dt,dt,{'Explicit':0,'Implicit':1,'CN':.5,'Rannacher':.5}[scheme])
        if n==0:first=u.copy()
        if not np.isfinite(u).all() or np.max(np.abs(u))>1e100:
            return dict(price=float('nan'),ds=ds,dt=dt,stability_ratio=dt*(SIG**2*(M-1)**2+R),min_all=float(u.min()),max_all=float(u.max()),first_min=float(first.min()),first_max=float(first.max()),first_local_negative_differences=int(np.sum(np.diff(first[(s>=125)&(s<=155)])<0))) if not snapshot else (dict(price=float('nan'),ds=ds,dt=dt,stability_ratio=dt*(SIG**2*(M-1)**2+R),min_all=float(u.min()),max_all=float(u.max())),s,first,u)
        minall=min(minall,float(u.min())); maxall=max(maxall,float(u.max()))
    result=dict(price=float(np.interp(S0,s,u)),ds=ds,dt=dt,
                stability_ratio=dt*(SIG**2*(M-1)**2+R),min_all=minall,max_all=maxall)
    if snapshot:return result,s,first,u
    return result

def timed(fun,repeats=3):
    durations=[]
    for _ in range(repeats):
        t=time.perf_counter(); result=fun(); durations.append(time.perf_counter()-t)
    return result,float(np.median(durations))

def barrier_sample(n,m,seed,bridge=False):
    rng=np.random.default_rng(seed); x=np.full(n,np.log(S0)); weight=np.ones(n); dt=T/m
    for _ in range(m):
        y=x+MU*dt+SIG*np.sqrt(dt)*rng.standard_normal(n)
        if bridge: weight*=bridge_weight(x,y,dt)
        else: weight*=(y<np.log(130))
        x=y
    return DISC*np.maximum(np.exp(x)-100,0)*weight

def barrier_paired(n,m,seed):
    rng=np.random.default_rng(seed); x=np.full(n,np.log(S0)); w=np.ones(n); alive=np.ones(n,dtype=bool); dt=T/m
    for _ in range(m):
        y=x+MU*dt+SIG*np.sqrt(dt)*rng.standard_normal(n)
        w*=bridge_weight(x,y,dt); alive &= y<np.log(130); x=y
    payoff=DISC*np.maximum(np.exp(x)-100,0)
    return payoff*alive,payoff*w

def main():
    exact=digital_exact(); bref,err,bcheck=barrier_ref()
    meta=dict(seed=BASE,parameters=dict(S0=S0,r=R,sigma=SIG,T=T,digital_K=140,barrier_K=100,H=130),digital_exact=exact,barrier_reference=bref,quadrature_error=err,barrier_bridge_quadrature=bcheck,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),timing='Median of three complete calls; excludes plotting/I-O; same machine.')
    print('References',exact,bref,flush=True)
    rows=[]
    for n in [10000,100000,1000000]:
        for mi,method in enumerate(['Standard','Antithetic','IS']):
            y,sec=timed(lambda:digital_sample(n,method,BASE+mi+n))
            rows.append(dict(method=method,N_evaluations=n,N_independent=len(y),**stats(y),seconds=sec,abs_error=abs(y.mean()-exact)))
    save('digital_mc',rows)
    rep=[]
    for n in [1000,4000,16000,64000]:
        for mi,method in enumerate(['Standard','Antithetic','IS']):
            estimates=[]
            for b in range(40):
                st=stats(digital_sample(n,method,BASE+5000000+n+100*mi+b)); estimates.append(st['price'])
                rep.append(dict(method=method,N_evaluations=n,replicate=b,**st,error=st['price']-exact))
    save('digital_replicates',rep)
    conv=[]
    for method in ['Standard','Antithetic','IS']:
        for n in [1000,4000,16000,64000]:
            group=[q for q in rep if q['method']==method and q['N_evaluations']==n]
            conv.append(dict(method=method,N=n,rmse=np.sqrt(np.mean([q['error']**2 for q in group])),coverage=np.mean([q['ci_low']<=exact<=q['ci_high'] for q in group])))
    save('digital_rmse',conv)
    frows=[]
    for kind,grids in [('digital',[100,200,400,800,1600]),('barrier',[130,260,520,1040])]:
        for M in grids:
            for scheme in ['Implicit','CN','Rannacher']:
                ans,sec=timed(lambda:fd(kind,M,M,scheme))
                ref=exact if kind=='digital' else bref
                frows.append(dict(option=kind,M=M,N=M,scheme=scheme,smoothing=True,**ans,seconds=sec,abs_error=abs(ans['price']-ref)))
            if kind=='digital':
                ans,sec=timed(lambda:fd(kind,M,M,'Rannacher',False))
                frows.append(dict(option=kind,M=M,N=M,scheme='Rannacher',smoothing=False,**ans,seconds=sec,abs_error=abs(ans['price']-exact)))
            if M<=520:
                N=int(np.ceil(T*(SIG**2*(M-1)**2+R)/.9))
                ans,sec=timed(lambda:fd(kind,M,N,'Explicit'))
                frows.append(dict(option=kind,M=M,N=N,scheme='Explicit',smoothing=True,**ans,seconds=sec,abs_error=abs(ans['price']-ref)))
    save('fd_convergence',frows)
    checks=[]
    for top,M in [(400,800),(600,1200),(800,1600)]:
        ans=fd('digital',M,800,'Rannacher',True,top)
        checks.append(dict(Smax=top,M=M,N=800,**ans,abs_error=abs(ans['price']-exact)))
    save('digital_domain',checks)
    diag=[]
    for scheme,N in [('Explicit',12000),('Explicit',9000),('CN',20),('Rannacher',20),('Implicit',20)]:
        ans,s,first,last=fd('digital',400,N,scheme,snapshot=True)
        diag.append(dict(scheme=scheme,M=400,N=N,**ans,first_min=float(first.min()),first_max=float(first.max()),first_local_negative_differences=int(np.sum(np.diff(first[(s>=125)&(s<=155)])<0))))
        if N==20:
            save('startup_'+scheme,[dict(S=float(si),value=float(vi)) for si,vi in zip(s,first)])
    save('fd_diagnostics',diag)
    brows=[]
    for m in [1,4,16,64,256]:
        n=200000; seed=BASE+800000+m
        raw,bb=barrier_paired(n,m,seed)
        _,tr=timed(lambda:barrier_sample(n,m,seed,False))
        _,tb=timed(lambda:barrier_sample(n,m,seed,True))
        diff=stats(raw-bb)
        for method,y,sec in [('Discrete',raw,tr),('Bridge',bb,tb)]:
            brows.append(dict(method=method,N=n,m=m,**stats(y),seconds=sec,error=float(y.mean()-bref),paired_gap=diff['price'],paired_gap_se=diff['se']))
    y,sec=timed(lambda:barrier_sample(1000000,1,BASE+900000,True))
    brows.append(dict(method='Bridge',N=1000000,m=1,**stats(y),seconds=sec,error=float(y.mean()-bref),paired_gap='',paired_gap_se=''))
    save('barrier_mc',brows)
    print('Core experiments complete',flush=True)
    assert min(q['abs_error'] for q in frows if q['option']=='barrier')<1e-4
    assert min(q['abs_error'] for q in frows if q['option']=='digital')<1e-5
    meta['checks']='Passed: bridge/reflection quadrature agreement; FD convergence against independent benchmarks.'
    extension(meta)
    (DATA/'metadata.json').write_text(json.dumps(meta,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.1))
    for method in ['Standard','Antithetic','IS']:
        q=[v for v in conv if v['method']==method]; axes[0].loglog([v['N'] for v in q],[v['rmse'] for v in q],'o-',label=method)
    axes[0].set(xlabel='Payoff evaluations',ylabel='RMSE (40 independent runs)');axes[0].legend()
    for smooth in [False,True]:
        q=[v for v in frows if v['option']=='digital' and v['scheme']=='Rannacher' and v['smoothing']==smooth]
        axes[1].loglog([v['M'] for v in q],[v['abs_error'] for v in q],'o-',label='Cell average' if smooth else 'Point payoff')
    axes[1].set(xlabel='Spatial intervals M (N=M)',ylabel='Absolute digital price error');axes[1].legend()
    fig.tight_layout();fig.savefig(FIG/'digital_convergence.pdf');fig.savefig(FIG/'digital_convergence.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,2.8))
    for method in ['Discrete','Bridge']:
        q=[v for v in brows if v['method']==method and v['N']==200000]
        ax.errorbar([v['m'] for v in q],[v['price'] for v in q],yerr=[1.96*v['se'] for v in q],fmt='o-',capsize=3,label=method+' MC')
    ax.axhline(bref,color='black',ls='--',label='Reflection benchmark');ax.set_xscale('log',base=4)
    ax.set(xlabel='Monitoring intervals m',ylabel='Up-and-out price');ax.legend();fig.tight_layout();fig.savefig(FIG/'barrier_monitoring.pdf');fig.savefig(FIG/'barrier_monitoring.png',dpi=160);plt.close(fig)
    print(json.dumps(meta,indent=2),flush=True)



def geometric_price(m=12,k=100):
    times=np.arange(1,m+1)*T/m
    mean=np.log(S0)+MU*times.mean()
    var=SIG**2*np.minimum.outer(times,times).mean()
    d2=(mean-np.log(k))/np.sqrt(var); d1=d2+np.sqrt(var)
    return float(DISC*(np.exp(mean+var/2)*ndtr(d1)-k*ndtr(d2)))

def asian_sample(n,seed,control=True,return_paths=False):
    rng=np.random.default_rng(seed); x=np.full(n,np.log(S0)); sa=np.zeros(n); sl=np.zeros(n)
    paths=[]
    for _ in range(12):
        x+=MU/12+SIG/np.sqrt(12)*rng.standard_normal(n)
        stock=np.exp(x); sa+=stock
        if control:sl+=x
        if return_paths:paths.append(stock[:1000].copy())
    a=DISC*np.maximum(sa/12-100,0)
    g=DISC*np.maximum(np.exp(sl/12)-100,0) if control else None
    if return_paths:return a,g,np.array(paths).T
    return a,g


def asian_beta(seed, n=20000):
    """Fit on an independent pilot; production payoffs must use another seed."""
    a,g=asian_sample(n,seed)
    variance=np.var(g,ddof=1)
    if variance<=0:raise ValueError('Pilot control has zero variance.')
    return float(np.cov(a,g,ddof=1)[0,1]/variance)


def asian_cholesky(n,seed):
    """Independent implementation: sample the joint log-price distribution directly."""
    rng=np.random.default_rng(seed)
    dates=np.arange(1,13)/12
    mean=np.log(S0)+MU*dates
    chol=np.linalg.cholesky(SIG**2*np.minimum.outer(dates,dates))
    arithmetic=[];geometric=[]
    for first in range(0,n,100000):
        log_prices=mean+rng.standard_normal((min(100000,n-first),12))@chol.T
        arithmetic.append(DISC*np.maximum(np.exp(log_prices).mean(axis=1)-100,0))
        geometric.append(DISC*np.maximum(np.exp(log_prices.mean(axis=1))-100,0))
    return np.concatenate(arithmetic),np.concatenate(geometric)


def extension(meta):
    """Extension B: does the geometric control repay its pilot and runtime cost?"""
    control_mean=geometric_price()
    pilot_seed=BASE+10000000
    beta=asian_beta(pilot_seed)
    rows=[]
    for n in [10000,100000,1000000]:
        seed=BASE+11000000+n
        (plain,_),plain_seconds=timed(lambda:asian_sample(n,seed,False))
        def controlled_run():
            b=asian_beta(pilot_seed)
            a,g=asian_sample(n,seed)
            return a-b*(g-control_mean)
        controlled,cv_seconds=timed(controlled_run)
        for method,y,seconds,pilot_n in [('Standard',plain,plain_seconds,0),('Geometric CV',controlled,cv_seconds,20000)]:
            rows.append(dict(method=method,N=n,pilot_N=pilot_n,**stats(y),seconds=seconds))
    save('asian_mc',rows)
    # Independent seeds AND a different path construction check the implementation.
    validation_beta=asian_beta(BASE+14000000)
    a,g=asian_cholesky(2000000,BASE+15000000)
    validation=stats(a-validation_beta*(g-control_mean))
    geometric_validation=stats(g)
    assert np.all(a>=g-1e-10)
    assert abs(geometric_validation['price']-control_mean)<5*geometric_validation['se']
    production=next(q for q in rows if q['method']=='Geometric CV' and q['N']==1000000)
    difference_se=np.hypot(production['se'],validation['se'])
    assert abs(production['price']-validation['price'])<5*difference_se
    # Refit the pilot independently in each repetition, including its uncertainty.
    replicates=[]
    for rep in range(80):
        rep_beta=asian_beta(BASE+16000000+2*rep)
        a,g=asian_sample(10000,BASE+16000001+2*rep)
        for method,y in [('Standard',a),('Geometric CV',a-rep_beta*(g-control_mean))]:
            replicates.append(dict(replicate=rep,method=method,N=10000,beta=rep_beta,**stats(y)))
    save('asian_replicates',replicates)
    summary=[]
    for method in ['Standard','Geometric CV']:
        group=[q for q in replicates if q['method']==method]
        summary.append(dict(method=method,replicates=80,N=10000,
            mean_price=float(np.mean([q['price'] for q in group])),
            across_run_sd=float(np.std([q['price'] for q in group],ddof=1)),
            rms_reported_se=float(np.sqrt(np.mean([q['se']**2 for q in group])))))
    save('asian_replication_summary',summary)
    # Exact covariance-sum formula equals the closed expression for monthly dates.
    expected_log_variance=SIG**2*T*13*25/(6*12**2)
    assert abs(expected_log_variance-SIG**2*np.minimum.outer(np.arange(1,13)/12,np.arange(1,13)/12).mean())<1e-15
    meta['parameters'].update(asian_K=100,asian_m=12)
    meta['geometric_exact']=control_mean
    meta['asian_beta']=beta
    meta['asian_validation_beta']=validation_beta
    meta['asian_validation']=validation
    meta['geometric_validation']=geometric_validation
    meta['extension']=dict(choice='B',question='Does a geometric control repay its pilot and runtime cost?',
        production_counts=[10000,100000,1000000],pilot_N=20000,replicates=80,
        validation_N=2000000,validation_method='Direct joint-normal log-price sampling using Cholesky factorisation')
    meta['timing']='Median of three complete calls; excludes plotting/I-O; same machine. Asian CV includes a fresh 20,000-path pilot in each timed call.'
    meta['checks']='Passed: core benchmark and FD checks; independent joint-normal Asian validation; geometric known mean; arithmetic >= geometric; geometric covariance formula.'
    print('Asian extension complete',json.dumps({k:meta[k] for k in ['geometric_exact','asian_beta','asian_validation','geometric_validation']},indent=2),flush=True)


if __name__=='__main__':
    import sys
    if '--extension-only' in sys.argv:
        metadata=json.loads((DATA/'metadata.json').read_text())
        extension(metadata)
        (DATA/'metadata.json').write_text(json.dumps(metadata,indent=2))
    else:main()
