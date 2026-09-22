# Certificate run record — three eight-dimensional local certificates

Binary: `bb9` (source `bb9.cpp`, compiled `g++ -O2 -march=native`).
Invocation: `./bb9 <slope> <eps> <node_cap> <table_h> <NSH> <shard> <wmin>`
with `node_cap = 6e8`, `table_h = 1/4096`, `NSH = 16`, `wmin = 1e-9`.

A shard reports `PROVED` only when its stack empties with `stack_left = 0` and
`HARD = 0`, i.e. every box in that shard's root set was discharged by a rigorous
lower bound. Any unresolved box, any terminal cell, or any node-cap hit yields
`INCOMPLETE`. No shard returned `INCOMPLETE`.

## Certificate A — s = 1/2, eps = 526/78125 = 0.0067328
16/16 shards PROVED. Logs: `s0.5_0.log` ... `s0.5_15.log`.
An earlier single-shard run of the same statement closed in 43,685,299 nodes / 198 s.

## Certificate B — s = 19/20, eps = 9879/1250000 = 0.0079032
16/16 shards PROVED. Nodes per shard:

    0:10742529  1:15696271  2:10042563  3:10046771
    4: 8803931  5:12217751  6:10388318  7: 9783922
    8:10222130  9:13039164 10:12851254 11: 8136142
   12: 6772742 13: 9987346 14:11895274 15:10575126
   total 170,801,234 nodes

## Certificate C — s = 1, eps = 20033/2500000 = 0.0080132
16/16 shards PROVED. Nodes per shard:

    0:13928410  1:15372708  2:14998246  3:14581026
    4:15414040  5:19415397  6:11970679  7:12710003
    8:12519853  9:16605643 10:12213013 11:13808935
   12:10506867 13: 8983453 14: 9061123 15:11430433
   total 213,519,829 nodes

## Reproduction
    ./run_all.sh          # replays all 48 shards and fails closed on any non-PROVED line
