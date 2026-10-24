# Audit of completed E0/E1 coverage before the new six-check cycle

E0= submission.py,source05922b7bf0353a3f4a54aaa27f9eef0692a4ee32 (onlyLOG2Efix),
H200A03bundlecf36a66/result84e54d6. E1= separate_merge.py,source
3818c7255cd2d9aa5859b2e55ee83f2bd2f057d4,H200A05bundle0524fdd/resulteb0c095.
E1full-only; slidingbotharms execute E0 unchanged, not E1source.

- Same graph/capturedworkbuffers changed in-place via copy_ for8differentseeds
  (2excludedwarmups+6steady), per12scoredshape/runner. No recapturebetweenreplays.
  data_ptr values were not logged; stablebufferuse follows executable restorecode.
- Explicit A->B->A was NOT run: all8seeds are distinct and none returns toA.
- B16/B32agenticcases regenerate actualGPUlengths,positions,cachelengths,page
  mappings/cache/qkv andcopythem into graphbuffers. B1fixedctxcases retain80000.
  Actuallengthvectors were NOT saved, so there is no independent numericlengthtrace.
- All16prerequisites:1checkedcapturedreplay perrunner after3eagerwarmupcalls.
  10uniqueprerequisites usepage64(default),2usekvscale1/64: slidingB3L8page1,
  fullB2L1page64. These parametercases did NOT get varied-contentgraphreplays.
  No page64+nonunitscale+L8 combinedcase. All12timedshapes page1/scale1.
- Output/newKV checked everycheckedreplay by unmodifiedofficialcollect/check:
  cache8%bound, outputadaptive3–15%plusrowrules. Allothercache rows and noncache
  inputs/config bytewiseunchangedchecked. Restorefromsnapshots beforeeacharm.
- Eachapp has32prerequisitechecks+192checkedwarmup/steadyreplays=224. E1actualnew
  fullfunction covers10fullprereqs and6fulltimedshapes; slidingisE0control.
- Sequentialgraphsonly; no concurrentstream/graphstress and no directticketvalue
  traces. PriorE0globalticketoverlap limitation remains; E1hasno sharedcounter.

0.9164023625864471 was recomputed read-only fromrawreceipts. For each of6loads,
eachrepetitionfirstsums46*sliding_us+16*full_us; adjacentoppositeordersform3
geometricpairratios v2/merge; takemedianperload thengeometricmeanof6loads.
It is NOT unweightedaveragingof12shaperatios, and NOT an officialscore/calibrated
judgevalue. Existing summaries/rawreceipts match exactly; no GPU/check/edit/push
wasused to obtain these auditfacts. Subsequentedits/newchecks require the later
authorization, separately journaled below.

Missingfocusedvalidation to add: explicit same-address A/B/A withactuallength
vectors logged/restored, pluspage64+scale1/64+L8. New GPUevidence mustbe labeled
separately and not retroactively attributed to E0/E1's original receipts.
