import threading
import time
import os

import pytest
import torch
from hivemind import DHT, get_logger

from flexgen.auto_config import AutoDistributedConfig
from flexgen.client_manager import RemoteSequenceManager, RemoteSequential
# from petals.data_structures import UID_DELIMITER
# from test_utils import *

logger = get_logger(__name__)

UID_DELIMITER = "."
#INITIAL_PEERS = os.environ.get("INITIAL_PEERS")
#if not INITIAL_PEERS:
#    raise RuntimeError("Must specify INITIAL_PEERS environment variable with one or more peer ids")
#INITIAL_PEERS = INITIAL_PEERS.split()

#MODEL_NAME = os.environ.get("MODEL_NAME")
#if not MODEL_NAME:
#    raise RuntimeError("Must specify MODEL_NAME as an index of a transformer block to be tested")

INITIAL_PEERS = [
    # IPv4 DNS addresses
    "ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
    "ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
    # Reserved IPs
    "/ip4/54.177.237.94/",
    "/ip4/52.53.152.100/"]

 #MODEL_NAME = os.environ.get("MODEL_NAME")
#if not MODEL_NAME:
#    raise RuntimeError("Must specify MODEL_NAME as an index of a transformer block to be tested")
MODEL_NAME = "bert"

#MODEL_NAME = os.environ.get("MODEL_NAME")
#if not MODEL_NAME:
#    raise RuntimeError("Must specify MODEL_NAME as an index of a transformer block to be tested")


@pytest.mark.forked
@pytest.mark.parametrize("mode", ["max_throughput", "min_latency"])
def test_sequence_manager_basics(mode: str):
    # how to replace?
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    sequential = RemoteSequential(config, dht=dht)
    shutdown_evt = threading.Event()

    # test RemoteSequential with lossy compression
    block_uids = [f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(config.num_hidden_layers)]
    sequential = RemoteSequential(
        config,
        sequence_manager=RemoteSequenceManagerWithChecks(config, block_uids, dht=dht, _was_shut_down=shutdown_evt),
    )

    sequence = sequential.sequence_manager.make_sequence(mode=mode)
    assert all(sequence[i].peer_id != sequence[i + 1].peer_id for i in range(len(sequence) - 1))

    assert sequential.sequence_manager.is_alive()
    assert sequential.sequence_manager._thread.ready.is_set()
    assert not shutdown_evt.is_set()
    sequential(torch.randn(1, 2, config.hidden_size))

    sequential.sequence_manager.shutdown()
    del sequential
    time.sleep(1)

    assert shutdown_evt.is_set()


class RemoteSequenceManagerWithChecks(RemoteSequenceManager):
    """A sequence manager that signals if it was shut down"""

    def __init__(self, *args, _was_shut_down: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._was_shut_down = _was_shut_down

    def shutdown(self):
        super().shutdown()
        assert not self.is_alive()
        self._was_shut_down.set()

if __name__ == "__main__":
    test_remote_sequential()
    