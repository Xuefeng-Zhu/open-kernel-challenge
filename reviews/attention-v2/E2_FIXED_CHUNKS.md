# E2 — fixed8192historical chunks, corrected-v2 parent

Source fixed_chunks.py,SHA7efbb370d2828effabdc7ed717d2e7d5785f9ee0da55380fad6e591adf7c8782. Onlyfullattentionbatch16/32 changes:
CHUNK_SIZE8192;SPLITS=ceil(shape-onlypage_tablecapacity/8192),scratch/grid sized
fromthatbound. Actualseq_lens staydevice-side. Chunks[split*8192,min(...,seq))
coverallhistoricalkeys exactlyonce. Emptychunkskeepneutralm=-inf/l=0/acc=0,
publishscratchandtickets; finalsplitstillownsallnewtokens evenifhistoryempty.
Allotherpaths use originalpartition,block64,widths,warps4,launchstages3,BF16/FP8/
FP32math, QKnorm/rope, gates andsamecompletionprotocol. E1notstacked.

TwoCPU/ASTinvariants+syntax pass: intervalcoverage/emptychunks atboundaries and
unchangedentireattention/newtoken/neutralpublicationbody. No GPUclaimyet. Add
focusedactualH200A/B/A atfixedaddresses, lengthvectorlogs, page64/scale1/64/L8
beforeofficialE2check. Wholeofficialcontractmuststillpass. PreserveoriginalE0.

E0accepted5b0baa286659421a,newbudget1/6,remaining5withonefinalreserved. GPU
Modalcumulativeguard1.565468698/2 unchanged; shortprobe mustreservewithinremaining
0.434531302, no newcredentials/paidcapchanges. This source is complete all-path
entrypoint, unlikehistoricalfull-onlyE1. No scoreclaimed fromCPUmodels.
