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
    'csnet_ns': 4,
    'ctf': 5,
    'daytime': 6,
    'discard': 7,
    "domain": 8,
    "domain_u": 9,
    "echo": 10,
    "eco_i": 11,
    "ecr_i": 12,
    "efs": 13,
    "exec": 14,
    "finger": 15,
    "ftp": 16,
    "ftp_data": 17,
    "gopher": 18,
    "harvest": 19,
    "hostnames": 20,
    "http": 21,
    "http_2784": 22,
    "http_443": 23,
    "http_8001": 24,
    "imap4": 25,
    "IRC": 26,
    "iso_tsap": 27,
    "klogin": 28,
    "kshell": 29,
    "ldap": 30,
    "link": 31,
    "login": 32,
    "mtp": 33,
    "name": 34,
    "netbios_dgm": 35,
    "netbios_ns": 36,
    "netbios_ssn": 37,
    "netstat": 38,
    "nnsp": 39,
    "nntp": 40,
    "ntp_u": 41,
    "other": 42,
    "pm_dump": 43,
    "pop_2": 44,
    "pop_3": 45,
    "printer": 46,
    "private": 47,
    "red_i": 48,
    "remote_job": 49,
    "rje": 50,
    "shell": 51,
    "smtp": 52,
    "sql_net": 53,
    "ssh": 54,
    "sunrpc": 55,
    "supdup": 56,
    "systat": 57,
    "telnet": 59,
    "tftp_u": 60,
    "tim_i": 61,
    "time": 62,
    "urh_i": 63,
    "urp_i": 64,
    "uucp": 65,
    "uucp_path": 66,
    "vmnet": 67,
    "whois": 68,
    "X11": 69,
    "Z39_50": 70


}

data['atr3'].replace(service, inplace=True)

#Flags

flags = {
    "OTH": 0,
    "REJ": 1,
    "RSTO": 2,
    "RSTOS0": 3,
    "RSTR": 4,
    "S0": 5,
    "S1": 6,
    "S2": 7,
    "S3": 8,
    "SF": 9,
    "SH": 10
}

data['atr4'].replace(flags, inplace=True)



print(data)
