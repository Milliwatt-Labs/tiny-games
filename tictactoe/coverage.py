"""Which complete games the game-play run could see: the union of every line each network can play (its full game
tree against every opponent reply, as X and as O) and of every line from the random openings of group C. Compared
with the 2,391 distinct games the GPU run recorded.

    python3 coverage.py        -> results/coverage.json
"""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.dirname(os.path.abspath(__file__))
import game as G, net as N
nets={k:N.load(D+'/weights/'+v) for k,v in {"finisher": "smallest-finisher.json", "satxo": "smallest-sat-xo.json", "xonly": "smallest-x-only.json"}.items()}
def lines(net, side, b=G.START, seq=()):
    if G.is_terminal(b): yield seq; return
    if side==0 or G.to_move(b)==side:
        m=N.choose_move(net,b); yield from lines(net,side,G.play(b,m),seq+(m,))
    else:
        for m in G.legal(b): yield from lines(net,side,G.play(b,m),seq+(m,))
U=set(); per={}
for k,n in nets.items():
    for side in ((G.X,G.O) if k!='xonly' else (G.X,)):
        L=set(lines(n,side)); per[f'{k}-{side}']=len(L); U|=L
AB=len(U)
for k in ('finisher','satxo'):
    for depth in (1,2,3):
        def op(b,seq,d):
            if d==0 or G.is_terminal(b):
                yield from lines(nets[k],0,b,seq); return
            for m in G.legal(b): yield from op(G.play(b,m),seq+(m,),d-1)
        L=set(op(G.START,(),depth)); per[f'C{depth}-{k}']=len(L); U|=L
hu = json.load(open(os.path.join(D, 'results', 'humanity.json')))
out = {'lines_per_tree': per, 'union_A_B': AB, 'union_all': len(U), 'gpu_distinct_games': hu['totals']['distinct_games_all_arms'],
       'gpu_distinct_per_arm': {a['name']: a['distinct_games'] for a in hu['arms']}}
json.dump(out, open(os.path.join(D, 'results', 'coverage.json'), 'w'), indent=1)
print(per, 'A+B union', AB, 'all', len(U), 'gpu', out['gpu_distinct_games'])
