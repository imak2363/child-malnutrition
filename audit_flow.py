import sys, logging
from decimal import Decimal
import numpy as np
import pandas as pd
from pygrowup import Calculator
 
logging.disable(logging.CRITICAL)
CALC = Calculator(adjust_height_data=False, adjust_weight_scores=False, include_cdc=False)
 
 
def z_or_nan(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return np.nan
 
 
def recompute(d):
    """WAZ, HAZ and WHZ from weight, height, age and sex (WHO standards, pygrowup 0.8.2)."""
    out = []
    for _, r in d.iterrows():
        sex = 'M' if str(r.SEX).lower().startswith('m') else 'F'
        age, w, h = Decimal(str(r.MONTH)), Decimal(str(r.WEIGHT)), Decimal(str(r.HEIGHT))
        out.append((z_or_nan(CALC.wfa, w, age, sex), z_or_nan(CALC.lhfa, h, age, sex),
                    z_or_nan(CALC.wfl if r.MONTH < 24 else CALC.wfh, w, age, sex, h)))
    return pd.DataFrame(out, columns=['waz', 'haz', 'whz'], index=d.index)
 
 
def who_flags(waz, haz, whz):
    fw, fh, fwh = (waz < -6) | (waz > 5), (haz < -6) | (haz > 6), (whz < -5) | (whz > 5)
    return fw, fh, fwh, fw | fh | fwh
 
 
def line(label, n, den=None):
    print(f"  {label:<78s} {n:>4d}" + (f" / {den:<4d} ({n / den * 100:.1f}%)" if den else ""))
 
 
def main(csv_path, sav_path, xlsx_path):
    import pyreadstat
    old = pd.read_csv(csv_path)
    sav, _ = pyreadstat.read_sav(sav_path)
    new = pd.read_excel(xlsx_path)
 
    cols = ['MONTH', 'WEIGHT', 'HEIGHT', 'MUAC']
    assert len(old) == len(sav) and np.isclose(old[cols].values, sav[cols].values).all(), \
        "the two files of the earlier version do not hold the same records in the same order"
    old['ID'] = sav.ID.values.astype(int)
    old['old_un'] = old.Result.str.lower().str.contains('under').astype(int)
    new['new_un'] = ((new.WAZ < -2) | (new.HAZ < -2) | (new.WHZ < -2)).astype(int)
    old = pd.concat([old, recompute(old)], axis=1)
    zn = recompute(new)
 
    print("Versions compared")
    line("Records in the earlier version", len(old))
    line("Records in the analysis version", len(new))
    line("Undernourished, earlier version (stored label)", int(old.old_un.sum()), len(old))
    line("Undernourished, analysis version (any Z-score below -2)", int(new.new_un.sum()), len(new))
    line("Difference (children)", int(old.old_un.sum() - new.new_un.sum()), len(old))
 
    m = old.merge(new, on='ID', suffixes=('_o', '_n'))
    same_id = (m.Area_o.str.lower() == m.Area_n.str.lower()) & \
              (m.SEX_o.str.lower().str[0] == m.SEX_n.str.lower().str[0]) & ((m.BOD_o % 100) == (m.BOD_n % 100))
    same_age = np.isclose(m.MONTH_o, m.MONTH_n, atol=0.02)
    print("\nRecord linkage")
    line("Same identifier, district, sex and birth year in the two versions", int(same_id.sum()), len(old))
    line("Age in months also agrees (linkage confirmed)", int((same_id & same_age).sum()), len(old))
    line("Age disagrees (linkage not confirmed)", int((same_id & ~same_age).sum()), len(old))
 
    L, U = m[same_id & same_age], m[same_id & ~same_age]
    dW, dH, dM = (~np.isclose(L.WEIGHT_o, L.WEIGHT_n), ~np.isclose(L.HEIGHT_o, L.HEIGHT_n), ~np.isclose(L.MUAC_o, L.MUAC_n))
    disc = dW | dH | dM
    dis = (L.old_un != L.new_un).values
    print(f"\nConfirmed records (n = {len(L)})")
    line("Weight, height and MUAC identical in the two versions", int((~disc).sum()), len(L))
    line("At least one of the three differs", int(disc.sum()), len(L))
    print(f"      weight differs in {int(dW.sum())}, height in {int(dH.sum())}, MUAC in {int(dM.sum())}")
    line("Outcome label agrees", int((~dis).sum()), len(L))
    line("Outcome label disagrees", int(dis.sum()), len(L))
    line("   undernourished in the earlier version only",
         int(((L.old_un == 1) & (L.new_un == 0)).sum()), int(dis.sum()))
    line("   of which among the discrepant records", int((dis & disc).sum()), int(disc.sum()))
    line("   of which among the identical records", int((dis & ~disc).sum()), int((~disc).sum()))
 
    print(f"\nUnconfirmed records (n = {len(U)})")
    line("Undernourished in the earlier version", int(U.old_un.sum()), len(U))
    line("Undernourished in the analysis version", int(U.new_un.sum()), len(U))
    print(f"\nReconciliation: {int(dis.sum())} + {int(U.old_un.sum() - U.new_un.sum())} = "
          f"{int(dis.sum() + U.old_un.sum() - U.new_un.sum())} label changes = "
          f"{int(old.old_un.sum())} - {int(new.new_un.sum())}")
 
    fw, fh, fwh, fany = who_flags(old.waz, old.haz, old.whz)
    print("\nWHO flags for implausible values")
    line(f"Earlier version, recomputed Z-scores (WAZ {int(fw.sum())}, HAZ {int(fh.sum())}, WHZ {int(fwh.sum())})",
         int(fany.sum()), len(old))
    fl_L = who_flags(L.waz, L.haz, L.whz)[3].values
    line("   of which among the discrepant confirmed records", int((fl_L & disc).sum()))
    line("   of which among the identical confirmed records", int((fl_L & ~disc).sum()))
    line("   of which among the unconfirmed records", int(who_flags(U.waz, U.haz, U.whz)[3].sum()))
    fw, fh, fwh, fany = who_flags(zn.waz, zn.haz, zn.whz)
    line(f"Analysis version, recomputed Z-scores (WAZ {int(fw.sum())}, HAZ {int(fh.sum())}, WHZ {int(fwh.sum())})",
         int(fany.sum()), len(new))
    line("Analysis version, supplied Z-scores (used in all analyses)",
         int(who_flags(new.WAZ, new.HAZ, new.WHZ)[3].sum()), len(new))
 
    def stored(col):
        t = col.str.lower()
        return np.where(t.str.contains('sev'), 'severe', np.where(t.str.contains('over'), 'over',
                        np.where(t.str.contains('normal'), 'normal', 'moderate')))
 
    def implied(z, has_over):
        return np.where(z < -3, 'severe', np.where(z < -2, 'moderate', np.where((z > 2) & has_over, 'over', 'normal')))
 
    # Stored categories against categories implied by the recomputed Z-scores: severe below -3,
    # moderate from -3 to below -2, overweight above +2 (WAZ and WHZ), otherwise normal.
    ok = ((stored(old.WAZ) == implied(old.waz, True)) & (stored(old.HAZ) == implied(old.haz, False)) &
          (stored(old.WHZ) == implied(old.whz, True)))
    # The stored WHZ labels have no overweight category; treating WHZ above +2 as normal instead:
    ok_lenient = ((stored(old.WAZ) == implied(old.waz, True)) & (stored(old.HAZ) == implied(old.haz, False)) &
                  (stored(old.WHZ) == implied(old.whz, False)))
    print("\nStored Z-score categories, earlier version")
    line("All three stored categories consistent with the recomputed Z-scores", int(ok.sum()), len(old))
    line("   the same, if WHZ above +2 is accepted as 'Normal'", int(ok_lenient.sum()), len(old))
 
    print("\nOther checks reported in Section 3.1")
    line("Height recorded as 64.0 cm, earlier version", int((old.HEIGHT == 64).sum()))
    line("Height recorded as 64.0 cm, analysis version", int((new.HEIGHT == 64).sum()))
    dup = old[['Area', 'SEX', 'BOD', 'MONTH', 'WEIGHT', 'HEIGHT', 'MUAC']].duplicated().sum()
    line("Duplicate records, earlier version (same district, sex, birth year, age and measurements)", int(dup))
    line("Duplicate identifiers, analysis version", int(new.ID.duplicated().sum()))
 
 
if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])