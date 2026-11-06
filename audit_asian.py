"""Final read-only numerical audit of the saved Asian experiment results."""
import csv,json
import numpy as np
import reproduce as m

def rows(name):
    with (m.DATA/(name+'.csv')).open() as f:return list(csv.DictReader(f))
def compare(expected,got):
    for key in ['price','se','variance','ci_low','ci_high']:
        assert np.isclose(float(expected[key]),got[key],rtol=1e-11,atol=1e-12),(key,expected[key],got[key])

meta=json.loads((m.DATA/'metadata.json').read_text())
for count in [1,2,12,52]:
    dates=np.arange(1,count+1)*m.T/count
    mean=np.log(m.S0)+m.MU*m.T*(count+1)/(2*count)
    var=m.SIG**2*m.T*(count+1)*(2*count+1)/(6*count**2)
    assert abs(mean-(np.log(m.S0)+m.MU*dates.mean()))<1e-14
    assert abs(var-m.SIG**2*np.minimum.outer(dates,dates).mean())<1e-14
assert abs(m.geometric_price()-6.990731415687477)<1e-12
beta=m.asian_beta(m.BASE+10000000)
assert abs(beta-meta['asian_beta'])<1e-12
table=rows('asian_mc'); metrics=[]
for n in [10000,100000,1000000]:
    a,g=m.asian_sample(n,m.BASE+11000000+n)
    assert np.all(a>=g-1e-12)
    plain=m.stats(a); control=m.stats(a-beta*(g-m.geometric_price()))
    p=next(r for r in table if r['method']=='Standard' and int(r['N'])==n)
    c=next(r for r in table if r['method']=='Geometric CV' and int(r['N'])==n)
    compare(p,plain);compare(c,control)
    vr=plain['variance']/control['variance']
    gain=vr*float(p['seconds'])/float(c['seconds'])
    metrics.append(dict(N=n,price=control['price'],se=control['se'],VR=vr,variance_time_gain=gain))
print('PASS: all six production rows, general lognormal moments, fixed pilot and known control mean.',flush=True)

vbeta=m.asian_beta(m.BASE+14000000)
a,g=m.asian_cholesky(2000000,m.BASE+15000000)
validation=m.stats(a-vbeta*(g-m.geometric_price()))
compare(meta['asian_validation'],validation)
compare(meta['geometric_validation'],m.stats(g))
z=abs(metrics[-1]['price']-validation['price'])/np.hypot(metrics[-1]['se'],validation['se'])
assert z<1
print('PASS: independent two-million-path joint-normal validation; difference < one combined SE.',flush=True)

saved=rows('asian_replicates'); results={'Standard':[],'Geometric CV':[]}
for rep in range(80):
    b=m.asian_beta(m.BASE+16000000+2*rep)
    a,g=m.asian_sample(10000,m.BASE+16000001+2*rep)
    for method,y in [('Standard',a),('Geometric CV',a-b*(g-m.geometric_price()))]:
        st=m.stats(y);results[method].append(st)
        row=next(r for r in saved if int(r['replicate'])==rep and r['method']==method)
        assert abs(b-float(row['beta']))<1e-12
        compare(row,st)
for row in rows('asian_replication_summary'):
    group=results[row['method']]
    assert np.isclose(np.std([v['price'] for v in group],ddof=1),float(row['across_run_sd']),rtol=1e-11)
    assert np.isclose(np.sqrt(np.mean([v['se']**2 for v in group])),float(row['rms_reported_se']),rtol=1e-11)
summary=dict(status='PASS',production_rows=6,replicate_rows=160,validation_paths=2000000,
             geometric_price=m.geometric_price(),beta=beta,production_metrics=metrics,
             independent_validation=validation,combined_se_difference=float(z),
             timing='Historical timings retained; variance-time gains recalculated from the saved timings.')
(m.ROOT/'docs'/'asian_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
