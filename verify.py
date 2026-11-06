"""Focused mathematical and application checks: python verify.py."""
import csv
import numpy as np
import reproduce as m
from app import calculate

def main():
    assert abs(m.digital_exact()-.0969151344528548)<1e-13
    reference,_,bridge_reference=m.barrier_ref()
    assert abs(reference-bridge_reference)<1e-11
    for kind,grids,ref in [('digital',[100,200,400],m.digital_exact()),
                           ('barrier',[130,260,520],reference)]:
        errors=[]
        for grid in grids:
            answer=m.fd(kind,grid,grid,'Rannacher')
            assert answer['min_all']>=-1e-12
            errors.append(abs(answer['price']-ref))
        assert all(3.5<errors[i]/errors[i+1]<4.5 for i in range(2))
    raw,bridge=m.barrier_paired(20000,16,20260914)
    assert np.all(raw>=bridge-1e-12)
    # A one-date geometric Asian must equal the vanilla Black--Scholes call.
    from scipy.special import ndtr
    d1=(m.R+.5*m.SIG**2)/m.SIG;d2=d1-m.SIG
    vanilla=m.S0*ndtr(d1)-100*m.DISC*ndtr(d2)
    assert abs(m.geometric_price(1)-vanilla)<1e-12
    a,g,paths=m.asian_sample(1000,20260914,True,True)
    assert paths.shape==(1000,12)
    np.testing.assert_allclose(a,m.DISC*np.maximum(paths.mean(axis=1)-100,0),atol=1e-12)
    np.testing.assert_allclose(g,m.DISC*np.maximum(np.exp(np.log(paths).mean(axis=1))-100,0),atol=1e-12)
    assert np.all(a>=g-1e-12)
    import json
    meta=json.loads((m.DATA/'metadata.json').read_text())
    validation=meta['asian_validation'];geometric=meta['geometric_validation']
    assert abs(geometric['price']-m.geometric_price())<5*geometric['se']
    with (m.DATA/'asian_mc.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    production=next(row for row in rows if row['method']=='Geometric CV' and row['N']=='1000000')
    assert abs(float(production['price'])-validation['price'])<5*np.hypot(float(production['se']),validation['se'])
    ref,asian_rows=calculate('Arithmetic Asian call',10000,100,12,130,123)
    assert ref is None and len(asian_rows)==2
    assert all(row['absolute_error']=='' for row in asian_rows)
    assert asian_rows[1]['se']<asian_rows[0]['se']/10
    for option in ['Digital call','Up-and-out call']:
        ref,rows=calculate(option,10000,140.,4,130,123)
        assert len(rows)==(7 if option=='Digital call' else 6)
        assert all(np.isfinite(row['price']) and row['price']>=0 for row in rows)
    print('PASS: analytical digital; independent barrier quadratures; second-order FD refinement;')
    print('bridge pathwise ordering; Asian dates/payoffs; geometric vanilla limit; independent Asian validation; all three application contracts.')

if __name__=='__main__':main()
