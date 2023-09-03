import pandapower.networks as nw
import pandapower as pp


fd=[0.85109443
    ,0.8232215
    ,0.8074705
    ,0.80322824
    ,0.81711454
    ,0.82330233
    ,0.82483813
    ,0.87451048
    ,0.93250569
    ,0.9629736
    ,0.9880899
    ,1
    ,0.98620011
    ,0.97703549
    ,0.99130643
    ,0.98018234
    ,0.97472204
    ,0.95226203
    ,0.95822405
    ,0.98511585
    ,0.98241775
    ,0.96047061
    ,0.95314281
    ,0.91203024]

net= nw.case14()

net.gen.loc[0,"vm_pu"]=1
Ybus = net._ppc["internal"]["Ybus"].todense()
pp.runpp(net)

perdidas=net.res_line["pl_mw"].sum()
abs(net.res_bus["vm_pu"]-1).sum()
net.load
print(perdidas)

nets=[]

T=24

for t in range(T):
    net=nw.case14()
    for i in range(11):
        net.load.loc[i,"scaling"]=fd[t]
    
    
    nets.append(net)
    
    pp.runpp(nets[t])
    perdidas=nets [t].res_line["pl_mw"].sum()
    print(perdidas)