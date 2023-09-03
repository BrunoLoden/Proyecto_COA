LU=[1.1,1.1,1.1,1.1,1.1,1.13636,0.806451,0.36764,0]
LL=[0.95,0.95,0.95,0.95,0.95,-1.13636,-0.806451,-0.36764,-20]

BU=LU*24
BL=LL*24


def FCIEEE14BUS (X):
    import pandapower.networks as nw
    import pandapower as pp
    
    net= nw.case14()
    net.ext_grid.loc[0,"vm_pu"]=X[0]
    net.gen.loc[0,"vm_pu"]=X[1]
    net.gen.loc[1,"vm_pu"]=X[2]
    net.gen.loc[2,"vm_pu"]=X[3]
    net.gen.loc[3,"vm_pu"]=X[4]
    net.trafo.loc[0,"tap_pos"]=X[5]
    net.trafo.loc[1,"tap_pos"]=X[6]
    net.trafo.loc[2,"tap_pos"]=X[7]
    net.shunt.loc[0,"q_mvar"]=X[8]
    pp.runpp(net)
    
    perdidas=net.res_line["pl_mw"].sum()
    return perdidas

