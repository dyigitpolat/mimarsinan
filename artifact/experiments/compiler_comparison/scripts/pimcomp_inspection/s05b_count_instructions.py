""
import sys,json,collections
W=sys.argv[1]
for m in ['regnet_y_400mf','regnet_y_800mf']:
    d=json.load(open(f'{W}/verif/VerificationInfo_batch_{m}.json'))
    nl=d['node_list']; cl=d['instruction']['core_list']
    per=collections.Counter(); ops=collections.Counter()
    for c in cl:
        for ins in (c or []):
            ops[ins['operation']]+=1
            if 'node_index' in ins: per[ins['node_index']]+=1
    byop=collections.defaultdict(lambda:[0,0])
    for i,n in enumerate(nl):
        byop[n['operation']][0]+=1; byop[n['operation']][1]+= (per[i]>0)
    print(m, {k:f'{v[1]} of {v[0]} nodes have instructions' for k,v in byop.items()}, 'total', sum(ops.values()), dict(ops))
