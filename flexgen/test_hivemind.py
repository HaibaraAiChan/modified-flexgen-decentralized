import hivemind

# INITIAL_PEERS = [
#     # IPv4 DNS addresses
#     "ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
#     "ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
#     # Reserved IPs
#     "/ip4/54.177.237.94/",
#     "/ip4/52.53.152.100/"]

# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     initial_peers=INITIAL_PEERS, start=True)

dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    start=True)

print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))