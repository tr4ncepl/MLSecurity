import numpy as np
import pandas as pd

data = pd.read_csv("dane.csv")
print(data)

#Classes

cl ={
    "anomaly": 1,
    "normal": 0
}
data['class'].replace(cl, inplace=True)




# Protocol type tcp, udp, icmp

protocol = {
    "tcp": 0,
    "udp": 1,
    "icmp": 2
}

data['atr2'].replace(protocol,inplace=True)

#Services

service = {
    "aol": 0,
    "auth": 1,
    "bgp": 2,
    "courier": 3,


}


print(data)
