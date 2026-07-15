"""
NAS100 OPTIMIZER — find the most profitable TP/SL + session for the VORTEX strategy.
Tests on real NAS100 5m data (histdata.com, Jan-Jun 2026, 25 weeks).
Focus: maximum profit per trade AND per week.
"""
import numpy as np, pandas as pd, itertools

CONTRACT = 20.0   # NAS100 CFD: $20 per point per lot
RISK_PCT = 0.025
START_BAL = 5000.0


def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr_calc(h,l,c,n=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def rsi_calc(s,n=14):
    d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+u/(dn+1e-12))


def build_features(df, entry_tf='5min', trend_tf='1h'):
    """Build features on the entry timeframe with HTF trend from trend_tf."""
    base = df.resample(entry_tf).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    d = base.copy()
    h,l,c = d['High'],d['Low'],d['Close']
    d['atr']=atr_calc(h,l,c,14)
    d['ema5']=ema(c,5); d['ema9']=ema(c,9); d['ema21']=ema(c,21); d['ema50']=ema(c,50)
    d['rsi']=rsi_calc(c,14)
    d['macd']=ema(c,12)-ema(c,26); d['macd_sig']=ema(d['macd'],9); d['macd_hist']=d['macd']-d['macd_sig']
    d['bb_mid']=c.rolling(20).mean(); d['bb_std']=c.rolling(20).std()
    d['bb_upper']=d['bb_mid']+2*d['bb_std']; d['bb_lower']=d['bb_mid']-2*d['bb_std']
    ht = df.resample(trend_tf).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    ht['e20']=ema(ht['Close'],20); ht['e50']=ema(ht['Close'],50)
    ht['trend']=np.where(ht['e20']>ht['e50'],1,np.where(ht['e20']<ht['e50'],-1,0))
    d['trend_ht']=ht['trend'].reindex(d.index, method='ffill').fillna(0)
    return d.dropna(subset=['atr','ema5','ema21','rsi','bb_mid'])


def gen_signals(d, sess_start, sess_end):
    sigs=[]
    for i in range(3,len(d)):
        row=d.iloc[i]; prev=d.iloc[i-1]; hour=row.name.hour
        if hour<sess_start or hour>sess_end: continue
        if row['atr']<0.01: continue
        direction=None
        cu = prev['ema5']<=prev['ema21'] and row['ema5']>row['ema21']
        cd = prev['ema5']>=prev['ema21'] and row['ema5']<row['ema21']
        if cu and row['trend_ht']>=0 and row['rsi']>45: direction='BUY'
        elif cd and row['trend_ht']<=0 and row['rsi']<55: direction='SELL'
        if direction is None:
            if row['rsi']>55 and prev['rsi']<50 and row['trend_ht']>=0 and row['Close']>row['ema21']: direction='BUY'
            elif row['rsi']<45 and prev['rsi']>50 and row['trend_ht']<=0 and row['Close']<row['ema21']: direction='SELL'
        if direction is None:
            if prev['Close']<=prev['bb_lower'] and row['Close']>row['bb_lower'] and row['trend_ht']>=0: direction='BUY'
            elif prev['Close']>=prev['bb_upper'] and row['Close']<row['bb_upper'] and row['trend_ht']<=0: direction='SELL'
        if direction is None:
            if prev['macd_hist']<0 and row['macd_hist']>0 and row['ema9']>row['ema21'] and row['trend_ht']>=0: direction='BUY'
            elif prev['macd_hist']>0 and row['macd_hist']<0 and row['ema9']<row['ema21'] and row['trend_ht']<=0: direction='SELL'
        if direction is None: continue
        sigs.append({'idx':i,'dir':direction,'atr':row['atr']})
    return sigs


def backtest(d, sigs, sl_mult, tp_ratio, cooldown, max_day, max_hold, bar_min):
    eq=START_BAL; peak=START_BAL; trades=[]; daily={}; tday={}; last_exit=-99
    for s in sigs:
        i=s['idx']
        if i+1>=len(d): continue
        if i-last_exit<cooldown: continue
        eb=d.iloc[i+1]; entry=eb['Open']; et=eb.name; dk=str(et.date())
        tday[dk]=tday.get(dk,0)
        if tday[dk]>=max_day: continue
        if daily.get(dk,0)<=-(0.05*eq): continue
        if eq<=peak*0.90: continue
        dd=(peak-eq)/peak if peak>0 else 0
        er = 0.012 if dd>=0.06 else (0.018 if dd>=0.035 else RISK_PCT)
        risk=eq*er; atr_v=s['atr']; sld=sl_mult*atr_v; tpd=tp_ratio*sld
        if s['dir']=='BUY': sl=entry-sld; tp=entry+tpd
        else: sl=entry+sld; tp=entry-tpd
        lots=max(0.01, round(risk/(sld*CONTRACT),2))
        outcome=None; bh=0
        for j in range(i+2, min(i+max_hold, len(d))):
            b=d.iloc[j]; bh+=1
            if s['dir']=='BUY':
                if b['Low']<=sl: outcome='SL'; break
                if b['High']>=tp: outcome='TP'; break
            else:
                if b['High']>=sl: outcome='SL'; break
                if b['Low']<=tp: outcome='TP'; break
        if outcome is None: outcome='TIMEOUT'; bh=max_hold-2; exit_px=d.iloc[min(i+max_hold-1,len(d)-1)]['Close']
        if outcome=='TP': pnl=lots*tpd*CONTRACT
        elif outcome=='SL': pnl=-(lots*sld*CONTRACT)
        else:
            exit_px=d.iloc[min(i+bh+1,len(d)-1)]['Close']
            pnl=(exit_px-entry if s['dir']=='BUY' else entry-exit_px)*lots*CONTRACT
        eq+=pnl; peak=max(peak,eq); daily[dk]=daily.get(dk,0)+pnl
        tday[dk]+=1; last_exit=i+bh+1
        trades.append({'pnl':pnl,'outcome':outcome,'bh':bh,'eq':eq})
    if not trades: return None
    t=pd.DataFrame(trades); n=len(t); net=t['pnl'].sum()
    wins=(t['pnl']>0).sum(); losses=(t['pnl']<=0).sum()
    wr=wins/n*100
    gp=t.loc[t['pnl']>0,'pnl'].sum(); gl=abs(t.loc[t['pnl']<0,'pnl'].sum())
    pf=gp/gl if gl>0 else 99
    eqv=t['eq'].values; pk=np.maximum.accumulate(np.concatenate([[START_BAL],eqv]))
    ddp=(pk[1:]-eqv).max()/pk[np.argmax(pk[1:]-eqv)]*100 if (pk[1:]-eqv).max()>0 else 0
    weeks=25.0
    return {'n':n,'per_trade':net/n,'net':net,'per_wk':net/weeks,'tpw':n/weeks,
            'wr':wr,'pf':pf,'ddp':ddp,'dur_h':t['bh'].mean()*bar_min/60}


if __name__=='__main__':
    df = pd.read_parquet('data/nas100_5m_2026.parquet')
    df['dt']=pd.to_datetime(df['dt']); df=df.set_index('dt').sort_index()
    df.columns=['Open','High','Low','Close']
    print(f'NAS100 5m: {len(df)} bars, {df.index.min()} -> {df.index.max()}\n')

    feat = build_features(df, '5min', '1h')
    print(f'Features (5m entry / 1h trend): {len(feat)} bars\n')

    # ===== SESSION TEST (fixed baseline SL/TP first) =====
    sessions = {
        'Asian (0-7)': (0,7), 'London (7-12)': (7,12), 'LN+NY overlap (12-16)': (12,16),
        'NY (13-20)': (13,20), 'London+NY (7-20)': (7,20), 'NY-full (14-21)': (14,21),
        'Full day (0-23)': (0,23),
    }
    print('='*90)
    print('SESSION TEST (baseline SL 3.5x / TP 1.5:1, cooldown 6, max 2/day, hold 60):')
    print('='*90)
    hdr = ('Session'.ljust(24)+'Trades'.ljust(8)+'T/wk'.ljust(7)+'Per-trade'.ljust(11)
           +'WR%'.ljust(7)+'PF'.ljust(6)+'$/wk'.ljust(9)+'DD%'.ljust(6)+'Dur')
    print(hdr)
    print('-'*90)
    def row_str(label, r):
        return (label.ljust(24) + str(r['n']).ljust(8)
                + '{:.1f}'.format(r['tpw']).ljust(7)
                + ('$' + '{:.2f}'.format(r['per_trade'])).ljust(11)
                + '{:.1f}'.format(r['wr']).ljust(7)
                + '{:.2f}'.format(r['pf']).ljust(6)
                + ('$' + '{:.1f}'.format(r['per_wk'])).ljust(9)
                + '{:.1f}'.format(r['ddp']).ljust(6)
                + '{:.1f}h'.format(r['dur_h']))

    sess_results={}
    for sname,(ss,se) in sessions.items():
        sg=gen_signals(feat,ss,se)
        r=backtest(feat,sg,3.5,1.5,6,2,60,5)
        if r:
            sess_results[sname]=(r,(ss,se))
            print(row_str(sname, r))
    best_wk = max(sess_results.items(), key=lambda x: x[1][0]['per_wk'])
    best_pt = max(sess_results.items(), key=lambda x: x[1][0]['per_trade'])
    print()
    print(f'  BEST SESSION by $/week:    {best_wk[0]}')
    print(f'  BEST SESSION by per-trade: {best_pt[0]}')

    # ===== FULL SL/TP SWEEP across sessions + timeframes =====
    print('\n' + '='*90)
    print('FULL SWEEP: SL x TP x SESSION x TIMEFRAME (only profitable, DD<10%, >=30 trades)')
    print('='*90)

    tf_configs = [('5min', 5, 60), ('15min', 15, 24)]  # (tf, bar_min, max_hold)
    test_sessions = {'London (7-12)':(7,12), 'LN+NY (12-16)':(12,16),
                     'NY (13-20)':(13,20), 'London+NY (7-20)':(7,20), 'Full (0-23)':(0,23)}

    all_results = []
    for tf, bar_min, max_hold in tf_configs:
        f = build_features(df, tf, '1h')
        for sname,(ss,se) in test_sessions.items():
            sg = gen_signals(f, ss, se)
            for sl_m in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                for tp_r in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
                    r = backtest(f, sg, sl_m, tp_r, 6, 2, max_hold, bar_min)
                    if r and r['n'] >= 30 and r['ddp'] <= 10.0 and r['net'] > 0:
                        r.update({'tf':tf, 'sess':sname, 'sl':sl_m, 'tp':tp_r})
                        all_results.append(r)

    if not all_results:
        print('  No profitable config found within 10% DD on NAS100.')
    else:
        # Top by per-trade
        by_pt = sorted(all_results, key=lambda x:-x['per_trade'])[:12]
        print('\nTOP 12 by PROFIT PER TRADE:')
        print('TF     Session            SL   TP    n    T/wk  Per-trade  WR%   PF    $/wk    DD%')
        print('-'*90)
        for r in by_pt:
            print(r['tf'].ljust(7)+r['sess'].ljust(19)
                  +'{:.1f}'.format(r['sl']).ljust(5)+'{:.2f}'.format(r['tp']).ljust(6)
                  +str(r['n']).ljust(5)+'{:.1f}'.format(r['tpw']).ljust(6)
                  +('$'+'{:.2f}'.format(r['per_trade'])).ljust(11)
                  +'{:.1f}'.format(r['wr']).ljust(6)+'{:.2f}'.format(r['pf']).ljust(6)
                  +('$'+'{:.1f}'.format(r['per_wk'])).ljust(8)+'{:.1f}'.format(r['ddp']))
        # Top by $/week
        by_wk = sorted(all_results, key=lambda x:-x['per_wk'])[:12]
        print('\nTOP 12 by $/WEEK:')
        print('TF     Session            SL   TP    n    T/wk  Per-trade  WR%   PF    $/wk    DD%')
        print('-'*90)
        for r in by_wk:
            print(r['tf'].ljust(7)+r['sess'].ljust(19)
                  +'{:.1f}'.format(r['sl']).ljust(5)+'{:.2f}'.format(r['tp']).ljust(6)
                  +str(r['n']).ljust(5)+'{:.1f}'.format(r['tpw']).ljust(6)
                  +('$'+'{:.2f}'.format(r['per_trade'])).ljust(11)
                  +'{:.1f}'.format(r['wr']).ljust(6)+'{:.2f}'.format(r['pf']).ljust(6)
                  +('$'+'{:.1f}'.format(r['per_wk'])).ljust(8)+'{:.1f}'.format(r['ddp']))
        bw = by_wk[0]
        print('\n' + '='*90)
        print('BEST OVERALL (max $/week, DD-safe):')
        print('  TF {} | Session {} | SL {}x | TP {}:1'.format(bw['tf'],bw['sess'],bw['sl'],bw['tp']))
        print('  ${:.2f}/trade | ${:.1f}/wk | {:.1f} trades/wk | {:.1f}% WR | PF {:.2f} | DD {:.1f}% | {:.1f}h'.format(
            bw['per_trade'], bw['per_wk'], bw['tpw'], bw['wr'], bw['pf'], bw['ddp'], bw['dur_h']))
        print('='*90)
