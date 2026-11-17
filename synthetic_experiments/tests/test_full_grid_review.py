import itertools
import numpy as np
import pandas as pd
from synthetic_experiments.full_grid_review import factorial_tables


def test_factorial_partition_and_matched_contrast():
    grid={'a':[-1,0,1],'b':[-1,0,1],'c':[-1,0,1]}
    rows=[]
    for a,b,c in itertools.product(*grid.values()):
        y=2*a+3*a*b+4*a*b*c
        rows.append(dict(a=a,b=b,c=c,method='test',child_state='test',centroid_shift_minutes=y,peak_shift_minutes=y))
    main,pairs,parts,contrasts=factorial_tables(pd.DataFrame(rows),grid)
    variance=4*2/3+9*(2/3)**2+16*(2/3)**3
    np.testing.assert_allclose(parts.main_pct,100*(4*2/3)/variance)
    np.testing.assert_allclose(parts.pair_pct,100*(9*(2/3)**2)/variance)
    np.testing.assert_allclose(parts.higher_order_pct,100*(16*(2/3)**3)/variance)
    np.testing.assert_allclose(parts[['main_pct','pair_pct','higher_order_pct']].sum(axis=1),100)
    np.testing.assert_allclose(main[main.factor.eq('a')].total_pct,100)
    np.testing.assert_allclose(contrasts[contrasts.factor.eq('a')].mean_minutes,4)
    np.testing.assert_allclose(contrasts[contrasts.factor.eq('b')].mean_minutes,0,atol=1e-14)


def test_near_constant_response_does_not_report_normalized_noise():
    grid={'a':[-1,0,1],'b':[-1,0,1]}
    rows=[dict(a=a,b=b,method='test',child_state='test',centroid_shift_minutes=a*1e-8,peak_shift_minutes=0.) for a,b in itertools.product(*grid.values())]
    main,pairs,parts,contrasts=factorial_tables(pd.DataFrame(rows),grid)
    assert main[['main_pct','total_pct']].isna().all().all()
    assert pairs.variance_pct.isna().all()
    assert parts[['main_pct','pair_pct','higher_order_pct']].isna().all().all()
    assert np.isfinite(contrasts.mean_minutes).all()
