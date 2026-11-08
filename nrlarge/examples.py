"""Small reproducible examples and synthetic capacity studies, never national data."""
import csv
import math
import numpy as np
from .model import Case, Node, Load, add_series, number, records


def unbalanced_example():
    nodes = [Node('source.'+p,'source',p,'SLACK',vm=0 if p == 'N' else 1,va=a)
             for p,a in zip('ABCN',(0,-120,120,0))]
    nodes += [Node('service.'+p,'service',p,vm=0 if p == 'N' else 1,va=a,
                   region='demo',feeder='F1') for p,a in zip('ABCN',(0,-120,120,0))]
    z = np.full((4,4),.002+.015j,dtype=complex)
    np.fill_diagonal(z,.03+.1j); z[3,3] = .08+.15j
    stamps = []
    add_series(stamps,'four-wire-line',['source.'+p for p in 'ABCN'],['service.'+p for p in 'ABCN'],z)
    loads = [Load('customer-'+p,'service.'+p,'service.N',d,q,.2,.3,.5,1,'customer-'+p,'F1','demo')
             for p,d,q in zip('ABC',(.3,.18,.25),(.1,.05,.08))]
    loads += [Load('delta-AB','service.A','service.B',.1,.02,nominal=math.sqrt(3),customer='delta-shop',feeder='F1',region='demo')]
    return Case(nodes,stamps,loads)


def synthetic(customers, memory_gib=4, restart=30):
    if customers < 1:
        raise ValueError('Positive synthetic customer count required')
    # Fail before allocating millions of Python objects. Network repeats a more
    # exact preflight once actual sparsity is known.
    estimate = customers*(2*(2500+16*(restart+10))+16*800+1800)
    if estimate > memory_gib*2**30:
        raise MemoryError('Synthetic-case preflight exceeds coordinator RAM budget')
    case = Case([Node('grid.'+p,'grid',p,'SLACK',vm=0 if p == 'N' else 1,va=a)
                 for p,a in zip('ABCN',(0,-120,120,0))],[],[],100)
    for first in range(0,customers,128):
        feeder = f'F{first//128}'
        case.nodes.extend(Node(feeder+'.'+p,feeder,p,vm=0 if p == 'N' else 1,va=a,region='synthetic',feeder=feeder)
                          for p,a in zip('ABCN',(0,-120,120,0)))
        z = np.full((4,4),.00001+.00002j,dtype=complex); np.fill_diagonal(z,.0001+.0004j)
        add_series(case.stamps,feeder,['grid.'+p for p in 'ABCN'],[feeder+'.'+p for p in 'ABCN'],z)
        for k in range(first,min(first+128,customers)):
            phase,angle = ('ABC'[k%3],(0,-120,120)[k%3]); customer = f'C{k}'
            case.nodes.extend([Node(customer+'.P',customer,phase,vm=1,va=angle,region='synthetic',feeder=feeder,customer=customer),
                               Node(customer+'.N',customer,'N',vm=0,region='synthetic',feeder=feeder,customer=customer)])
            add_series(case.stamps,'service-'+customer,[feeder+'.'+phase,feeder+'.N'],[customer+'.P',customer+'.N'],
                       np.array([[.02+.04j,.002+.005j],[.002+.005j,.03+.04j]]))
            mw = .003+(k%11)*.0005
            case.loads.append(Load(customer,customer+'.P',customer+'.N',mw/(case.base_mva/3),.3*mw/(case.base_mva/3),
                                   .2,.3,.5,1,customer,feeder,'synthetic'))
    return case


def convert_balanced(buses, branches, base_mva=100):
    nodes = []
    for b in records(buses,'id,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu'.split(',')):
        if not b['id']:
            raise ValueError('Legacy bus IDs must be nonempty')
        for phase,offset in zip('ABC',(0,-120,120)):
            nodes.append(Node(b['id']+'.'+phase,b['id'],phase,b['type'],number(b['p_pu']),number(b['q_pu']),
                              number(b['vm_pu']),number(b['va_deg'])+offset,
                              number(b['qmin_pu']) if b['qmin_pu'] else -math.inf,
                              number(b['qmax_pu']) if b['qmax_pu'] else math.inf))
    stamps = []
    for b in records(branches,'id,from,to,r_pu,x_pu,b_pu,tap,shift_deg'.split(',')):
        magnitude = number(b['tap'])
        tap = magnitude * np.exp(1j*np.deg2rad(number(b['shift_deg'])))
        if magnitude <= 0 or number(b['r_pu']) < 0:
            raise ValueError('Invalid legacy tap/resistance')
        add_series(stamps,b['id'],[b['from']+'.'+p for p in 'ABC'],[b['to']+'.'+p for p in 'ABC'],
                   np.eye(3)*complex(number(b['r_pu']),number(b['x_pu'])),
                   np.eye(3)*1j*number(b['b_pu']),[tap]*3)
    return Case(nodes,stamps,[],base_mva)
