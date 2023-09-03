import math as m
Uk=500
Us=525


mc=0.95

tiempo="LLuvioso"

if tiempo == "LLuvioso":
    ma=0.8
else:
    ma=1


DM=0
Vr=0
#Conductor
diamABC=25.38
diamG=25.38
lonA=9
msnm=2500
tm=25-msnm/200
Subcon=2
Dissc=0.45
r1=diamABC/2*10**(-3)+Vr

class Conductor:
  def __init__(self,x,y):
    self.x = x
    self.y = y


def dis(p1,p2):
  return m.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

if Subcon==1:
    Con_A=Conductor(-7,41-lonA)
    Con_B=Conductor(7,35.5-lonA)
    Con_C=Conductor(-7,30-lonA)
    
    RMG_A=r1
    RMG_B=r1
    RMG_C=r1
    
    RMG_total=pow(RMG_A*RMG_B*RMG_C,1/3)
    
    DMG_AB=dis(Con_A,Con_B)
    DMG_AC=dis(Con_A,Con_C)
    DMG_BC=dis(Con_B,Con_C)
    
    
    DMG_total=pow(DMG_AB*DMG_AC*DMG_BC,1/3)

else:
  if Subcon==2:
    Con_A=Conductor(-6-Dissc/2-DM,41-lonA)
    Con_A_prima=Conductor(-6+Dissc-DM/2,41-lonA)
    Con_B=Conductor(8-Dissc/2+DM,35.5-lonA)
    Con_B_prima=Conductor(8+Dissc/2+DM,35.5-lonA)
    Con_C=Conductor(-6-Dissc/2-DM,30-lonA)
    Con_C_prima=Conductor(-6+Dissc/2-DM,30-lonA)

    RMG_A=pow(r1*Dissc,1/2)
    RMG_B=pow(r1*Dissc,1/2)
    RMG_C=pow(r1*Dissc,1/2)

    RMG_A_Cap=pow(diamABC/2*10**(-3)*Dissc,1/2)
    RMG_B_Cap=pow(diamABC/2*10**(-3)*Dissc,1/2)
    RMG_C_Cap=pow(diamABC/2*10**(-3)*Dissc,1/2)
    
    RMG_Capa=pow(RMG_A_Cap*RMG_B_Cap*RMG_C_Cap,1/3)

    RMG_total=pow(RMG_A*RMG_B*RMG_C,1/3)

    DMG_AB=pow(dis(Con_A,Con_B)*dis(Con_A,Con_B_prima)*dis(Con_A_prima,Con_B_prima)*dis(Con_A_prima,Con_B),1/4)
    DMG_AC=pow(dis(Con_A,Con_C)*dis(Con_A,Con_C_prima)*dis(Con_A_prima,Con_C)*dis(Con_A_prima,Con_C_prima),1/4)
    DMG_BC=pow(dis(Con_B,Con_C)*dis(Con_B,Con_C_prima)*dis(Con_B_prima,Con_C)*dis(Con_B_prima,Con_C_prima),1/4)


    DMG_total=pow(DMG_AB*DMG_AC*DMG_BC,1/3)

  else:
      Con_A=Conductor(-7,41-lonA-Dissc/m.sqrt(3))
      Con_A_prima1=Conductor(-7-Dissc/2,41-lonA+Dissc/m.sqrt(3)/2)
      Con_A_prima2=Conductor(-7+Dissc/2,41-lonA+Dissc/m.sqrt(3)/2)
      Con_B=Conductor(7,35.5-lonA-Dissc/m.sqrt(3))
      Con_B_prima1=Conductor(7-Dissc/2,35.5-lonA+Dissc/m.sqrt(3)/2)
      Con_B_prima2=Conductor(7+Dissc/2,35.5-lonA+Dissc/m.sqrt(3)/2)
      Con_C=Conductor(-7,30-lonA-Dissc/m.sqrt(3))
      Con_C_prima1=Conductor(-7-Dissc/2,30-lonA+Dissc/m.sqrt(3)/2)
      Con_C_prima2=Conductor(-7+Dissc/2,30-lonA+Dissc/m.sqrt(3)/2)
    
      RMG_A=pow(r1*Dissc**2,1/3)
      RMG_B=pow(r1*Dissc**2,1/3)
      RMG_C=pow(r1*Dissc**2,1/3)
    

      RMG_total=pow(RMG_A*RMG_B*RMG_C,1/3)
    
      DMG_AB=pow(dis(Con_A,Con_B)*dis(Con_A,Con_B_prima1)*dis(Con_A,Con_B_prima2)*dis(Con_A_prima1,Con_B)*dis(Con_A_prima1,Con_B_prima1)*dis(Con_A_prima1,Con_B_prima2)*dis(Con_A_prima2,Con_B)*dis(Con_A_prima2,Con_B_prima1)*dis(Con_A_prima2,Con_B_prima2),1/9)
      DMG_AC=pow(dis(Con_A,Con_C)*dis(Con_A,Con_C_prima1)*dis(Con_A,Con_C_prima2)*dis(Con_A_prima1,Con_C)*dis(Con_A_prima1,Con_C_prima1)*dis(Con_A_prima1,Con_C_prima2)*dis(Con_A_prima2,Con_C)*dis(Con_A_prima2,Con_C_prima1)*dis(Con_A_prima2,Con_C_prima2),1/9)
      DMG_BC=pow(dis(Con_B,Con_C)*dis(Con_B,Con_C_prima1)*dis(Con_B,Con_C_prima2)*dis(Con_B_prima1,Con_C)*dis(Con_B_prima1,Con_C_prima1)*dis(Con_B_prima1,Con_C_prima2)*dis(Con_B_prima2,Con_C)*dis(Con_B_prima2,Con_C_prima1)*dis(Con_B_prima2,Con_C_prima2),1/9)
    
    
      DMG_total=pow(DMG_AB*DMG_AC*DMG_BC,1/3)







delta=(273+25)/(273+tm)*m.exp(-msnm/8150)

Uc=30/m.sqrt(2)*mc*ma*(delta)*RMG_total*m.log(DMG_total/RMG_total)*100
Us/m.sqrt(3)

print(Uc)
f=60
Pcorona=241/delta*(f+25)*(Us/m.sqrt(3)-Uc)**2*m.sqrt(RMG_total/DMG_total)*10**(-5)
Pcorona*3