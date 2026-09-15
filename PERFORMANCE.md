# Performance items (not blocking, recorded so they are not lost)

## Payload on the grouped shared-gather tier

The grouped kernel (`genkernel.GroupedRowTerm`) fuses *payload-free* operators that sit on an
identical point set into one shared jet gather. Operators whose coefficients ride the per-row
payload (`cix >= 0`) — per-point frame coefficients, spatially varying `nu(x)`, per-row
covectors — are excluded and run on the per-operator kernel, one gather per row.

Cost today (cardiac in vivo recipe, acquisition-frame channels): the polar continuity and the two
polar momentum operators share 200k collocation points but carry six frame coefficients in the
payload, so they are gathered three times instead of once. About 6% of the rows of that system.

Fix: let the grouped kernel read a per-POINT payload (all co-located operators of one point share
it, which is exactly the case that arises: one frame per point), i.e. `bind_points(..., C=None)`
with `c[cix]` sourced from a point-indexed buffer in the grouped prelude. Verify with
`verify_grouped` as for the payload-free case.

## Wrapped operators and the grouped tier — NOT an item

Wrapped rows (modulus `m > 0`) run on the per-operator kernel. That is their natural tier: a
Doppler measurement is one operator on its own points and has no co-located partner to share a
gather with, so the grouped tier could never have served it. The modulus costs one int per row
record and three exponentials per row against a full jet gather. Only if a wrapped operator is ever
authored on a shared point set with other operators would a modulus on the grouped tier matter.
