import itertools

from IProcessor import IPosition, ParameterSpec, Positions

import numpy as np

class BL(IPosition):
    def __init__(self):
        super().__init__("BL", fullname="Bilateration", minGraphs=2)

    def DefineDevParams(self):
        return [
            ParameterSpec("PosX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosX, tunable=False),
            ParameterSpec("PosY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosY, tunable=False),
            ParameterSpec("NormX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealNormX, tunable=False),
            ParameterSpec("NormY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealNormY, tunable=False)
        ]
    
    def _Base(self, distances, devParams, params):
        pairs = list(itertools.combinations(range(distances.shape[0]), 2))
        Iidx  = np.array([ i for i,j in pairs ], dtype=int) # (P,)
        Jidx  = np.array([ j for i,j in pairs ], dtype=int) # (P,)
        distA = distances[Iidx]                             # (P,T)
        distB = distances[Jidx]                             # (P,T)
        posA  = devParams[Iidx,0:2]                         # (P,2)
        posB  = devParams[Jidx,0:2]                         # (P,2)
        normA = devParams[Iidx,2:4]                         # (P,2)
        normB = devParams[Jidx,2:4]                         # (P,2)

        dVec = posB - posA                    # (P,2)
        d    = np.linalg.norm(dVec, axis=1)   # (P,)
        dd   = np.clip(d, min=1e-9, max=None) # (P,)
        dxdy = dVec / dd[:,None]              # (P,2)

        a = (distA**2 - distB**2 + (dd**2)[:,None]) / (2 * dd[:,None]) # (P,T)
        h = np.sqrt(np.clip(distA**2 - a**2, min=0.0, max=None))       # (P,T)

        dx = dxdy[:,0] # (P,)
        dy = dxdy[:,1] # (P,)

        Px = posA[:,0][:,None] + a * dx[:,None]      # (P,T)
        Py = posA[:,1][:,None] + a * dy[:,None]      # (P,T)
        fx = h * dy[:,None]                          # (P,T)
        fy = h * dx[:,None]                          # (P,T)
        P1 = np.stack([ Px + fx, Py - fy ], axis=-1) # (P,T,2)
        P2 = np.stack([ Px - fx, Py + fy ], axis=-1) # (P,T,2)

        avgNorm = 0.5 * (normA + normB)                   # (P,2)
        dot     = np.sum(P1 * avgNorm[:,None,:], axis=-1) # (P,T)
        bestPos = np.where(dot[...,None] >= 0, P1, P2)    # (P,T,2)

        valid    = (d != 0)[:,None] & (h > 0)                                              # (P,T)
        dists    = np.linalg.norm(bestPos[None,...] - devParams[:,None,None,0:2], axis=-1) # (N,P,T)
        errs     = np.abs(dists - distances[:,None,:])                                     # (N,P,T)
        totalErr = np.where(valid, np.sum(errs, axis=0), np.inf)                           # (P,T)

        bestPairIdx  = np.argmin(totalErr, axis=0)      # (T,)
        timeIdx      = np.arange(bestPairIdx.shape[0])  # (T,)
        bestTotalErr = totalErr[bestPairIdx,timeIdx]    # (T,)
        validMask    = np.isfinite(bestTotalErr)        # (T,)
        finalPos     = bestPos[bestPairIdx, timeIdx, :] # (T,2)
        return (finalPos, validMask)
Positions().Define(BL)

class TLG(IPosition):
    def __init__(self):
        super().__init__("TLG", fullname="Geometric Trilateration", minGraphs=3)

    def DefineDevParams(self):
        return [
            ParameterSpec("PosX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosX, tunable=False),
            ParameterSpec("PosY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosY, tunable=False)
        ]
    
    def _Base(self, distances, devParams, params):
        triples = list(itertools.combinations(range(distances.shape[0]), 3))
        Iidx    = np.array([ i for i,j,k, in triples ], dtype=int) # (P,)
        Jidx    = np.array([ j for i,j,k, in triples ], dtype=int) # (P,)
        Kidx    = np.array([ k for i,j,k, in triples ], dtype=int) # (P,)
        distA   = distances[Iidx]                                  # (P,T)
        distB   = distances[Jidx]                                  # (P,T)
        distC   = distances[Kidx]                                  # (P,T)
        posA    = devParams[Iidx,0:2]                              # (P,2)
        posB    = devParams[Jidx,0:2]                              # (P,2)
        posC    = devParams[Kidx,0:2]                              # (P,2)

        ba = posB - posA                  # (P,2)
        ca = posC - posA                  # (P,2)
        cb = posC - posB                  # (P,2)
        lba = np.linalg.norm(ba, axis=-1) # (P,)
        lca = np.linalg.norm(ca, axis=-1) # (P,)
        lcb = np.linalg.norm(cb, axis=-1) # (P,)

        ba = ba / np.clip(lba, min=1e-9, max=None)[:,None] # (P,2)
        ca = ca / np.clip(lca, min=1e-9, max=None)[:,None] # (P,2)
        cb = cb / np.clip(lcb, min=1e-9, max=None)[:,None] # (P,2)

        Q1 = 0.5 * (posA + posB)[:,None,:] + 0.5 * (distA - distB)[:,:,None] * ba[:,None,:] # (P,T,2)
        Q2 = 0.5 * (posA + posC)[:,None,:] + 0.5 * (distA - distC)[:,:,None] * ca[:,None,:] # (P,T,2)
        n1 = np.stack((-ba[:,1], ba[:,0]), axis=-1) # (P,2)
        n2 = np.stack((-ca[:,1], ca[:,0]), axis=-1) # (P,2)

        div1 = n1[:,0] * n2[:,1] - n1[:,1] * n2[:,0] # (P,)

        t1   = (n2[:,0][:,None] * (Q1[:,:,1] - Q2[:,:,1]) - n2[:,1][:,None] * (Q1[:,:,0] - Q2[:,:,0])) / np.where(div1 == 0.0, 1e-9, div1)[:,None] # (P,T)

        pos = Q1 + t1[...,None] * n1[:,None,:] # (P,T,2)

        valid    = div1 != 0.0                                                         # (P,)
        dists    = np.linalg.norm(pos[None,...] - devParams[:,None,None,0:2], axis=-1) # (N,P,T)
        errs     = np.abs(dists - distances[:,None,:])                                 # (N,P,T)
        totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)               # (P,T)

        bestPairIdx  = np.argmin(totalErr, axis=0)     # (T,)
        timeIdx      = np.arange(bestPairIdx.shape[0]) # (T,)
        bestTotalErr = totalErr[bestPairIdx,timeIdx]   # (T,)
        validMask    = np.isfinite(bestTotalErr)       # (T,)
        finalPos     = pos[bestPairIdx,timeIdx,:]      # (U,2)
        return (finalPos, validMask)
Positions().Define(TLG)

class TLMT(IPosition):
    def __init__(self):
        super().__init__("TLMT", fullname="Trilateration Matrix Transform", minGraphs=3)

    def DefineDevParams(self):
        return [
            ParameterSpec("PosX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosX, tunable=False),
            ParameterSpec("PosY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosY, tunable=False)
        ]
    
    def _Base(self, distances, devParams, params):
        triples = list(itertools.combinations(range(distances.shape[0]), 3))
        Iidx    = np.array([ i for i,j,k, in triples ], dtype=int) # (P,)
        Jidx    = np.array([ j for i,j,k, in triples ], dtype=int) # (P,)
        Kidx    = np.array([ k for i,j,k, in triples ], dtype=int) # (P,)
        distA   = distances[Iidx]                                  # (P,T)
        distB   = distances[Jidx]                                  # (P,T)
        distC   = distances[Kidx]                                  # (P,T)
        posA    = devParams[Iidx,0:2]                              # (P,2)
        posB    = devParams[Jidx,0:2]                              # (P,2)
        posC    = devParams[Kidx,0:2]                              # (P,2)

        adota = np.sum(posA * posA, axis=-1)                  # (P,)
        bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
        cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
        AB1   = 2 * (posB - posA)                             # (P,2)
        AB2   = 2 * (posC - posA)                             # (P,2)
        AB3   = 2 * (posC - posB)                             # (P,2)
        C1    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
        C2    = distA**2 - distC**2 + (cdotc - adota)[:,None] # (P,T)
        C3    = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)

        a13 = AB1**2 + AB2**2 + AB3**2                                        # (P,2)
        a2  = AB1[:,0] * AB1[:,1] + AB2[:,0] * AB2[:,1] + AB3[:,0] * AB3[:,1] # (P,)

        det = a13[:,0] * a13[:,1] - a2**2 # (P,)

        c31 = a13 / np.where(det == 0.0, 1e-9, det)[:,None] # (P,2)
        c2  = -a2 / np.where(det == 0.0, 1e-9, det)         # (P,)

        b = AB1[:,None,:] * C1[...,None] + AB2[:,None,:] * C2[...,None] + AB3[:,None,:] * C3[...,None] # (P,T,2)

        pos = np.stack((c31[:,None,1] * b[:,:,0] + c2[:,None] * b[:,:,1], c2[:,None] * b[:,:,0] + c31[:,None,0] * b[:,:,1]), axis=-1) # (P,T,2)

        valid    = det != 0.0                                                          # (P,)
        dists    = np.linalg.norm(pos[None,...] - devParams[:,None,None,0:2], axis=-1) # (N,P,T)
        errs     = np.abs(dists - distances[:,None,:])                                 # (N,P,T)
        totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)               # (P,T)

        bestPairIdx  = np.argmin(totalErr, axis=0)     # (T,)
        timeIdx      = np.arange(bestPairIdx.shape[0]) # (T,)
        bestTotalErr = totalErr[bestPairIdx,timeIdx]   # (T,)
        validMask    = np.isfinite(bestTotalErr)       # (T,) 
        finalPos     = pos[bestPairIdx,timeIdx,:]      # (U,2)
        return (finalPos, validMask)
Positions().Define(TLMT)

class TLMD(IPosition):
    def __init__(self):
        super().__init__("TLMD", fullname="Trilateration Matrix Determinant", minGraphs=3)

    def DefineDevParams(self):
        return [
            ParameterSpec("PosX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosX, tunable=False),
            ParameterSpec("PosY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosY, tunable=False)
        ]
    
    def _Base(self, distances, devParams, params):
        triples = list(itertools.combinations(range(distances.shape[0]), 3))
        Iidx    = np.array([ i for i,j,k, in triples ], dtype=int) # (P,)
        Jidx    = np.array([ j for i,j,k, in triples ], dtype=int) # (P,)
        Kidx    = np.array([ k for i,j,k, in triples ], dtype=int) # (P,)
        distA   = distances[Iidx]                                  # (P,T)
        distB   = distances[Jidx]                                  # (P,T)
        distC   = distances[Kidx]                                  # (P,T)
        posA    = devParams[Iidx,0:2]                              # (P,2)
        posB    = devParams[Jidx,0:2]                              # (P,2)
        posC    = devParams[Kidx,0:2]                              # (P,2)

        a12 = 2 * (posB - posA)                         # (P,2)
        a34 = 2 * (posC - posB)                         # (P,2)
        ad  = a12[:,0] * a34[:,1] - a12[:,1] * a34[:,0] # (P,)

        adota = np.sum(posA * posA, axis=-1)                  # (P,)
        bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
        cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
        b1    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
        b2    = 2 * (posB[:,1] - posA[:,1])                   # (P,)
        b3    = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)
        b4    = 2 * (posC[:,1] - posB[:,1])                   # (P,)
        bd    = b1 * b4[:,None] - b2[:,None] * b3             # (P,T)

        c1    = 2 * (posB[:,0] - posA[:,0])                   # (P,)
        c2    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
        c3    = 2 * (posC[:,0] - posB[:,0])                   # (P,)
        c4    = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)
        cd    = c1[:,None] * c4 - c2 * c3[:,None]             # (P,T)

        pos = np.stack((bd, cd), axis=-1) / np.where(ad == 0.0, 1e-9, ad)[:,None,None] # (P,T,2)

        valid    = ad != 0.0                                                           # (P,)
        dists    = np.linalg.norm(pos[None,...] - devParams[:,None,None,0:2], axis=-1) # (N,P,T)
        errs     = np.abs(dists - distances[:,None,:])                                 # (N,P,T)
        totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)               # (P,T)

        bestPairIdx  = np.argmin(totalErr, axis=0)     # (T,)
        timeIdx      = np.arange(bestPairIdx.shape[0]) # (T,)
        bestTotalErr = totalErr[bestPairIdx,timeIdx]   # (T,)
        validMask    = np.isfinite(bestTotalErr)       # (T,) 
        finalPos     = pos[bestPairIdx,timeIdx,:]      # (U,2)
        return (finalPos, validMask)
Positions().Define(TLMD)

class TLSE(IPosition):
    def __init__(self):
        super().__init__("TLSE", fullname="Trilateration System of Equations", minGraphs=3)

    def DefineDevParams(self):
        return [
            ParameterSpec("PosX", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosX, tunable=False),
            ParameterSpec("PosY", default=lambda addr,sessionParameters:sessionParameters.Beacons[addr].RealPosY, tunable=False)
        ]
    
    def _Base(self, distances, devParams, params):
        triples = list(itertools.combinations(range(distances.shape[0]), 3))
        Iidx    = np.array([ i for i,j,k, in triples ], dtype=int) # (P,)
        Jidx    = np.array([ j for i,j,k, in triples ], dtype=int) # (P,)
        Kidx    = np.array([ k for i,j,k, in triples ], dtype=int) # (P,)
        distA   = distances[Iidx]                                  # (P,T)
        distB   = distances[Jidx]                                  # (P,T)
        distC   = distances[Kidx]                                  # (P,T)
        posA    = devParams[Iidx,0:2]                              # (P,2)
        posB    = devParams[Jidx,0:2]                              # (P,2)
        posC    = devParams[Kidx,0:2]                              # (P,2)

        adota = np.sum(posA * posA, axis=-1)                  # (P,)
        bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
        cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
        A  = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
        BC = 2 * (posB - posA) # (P,2)
        D  = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)
        EF = 2 * (posC - posB) # (P,2)

        div = BC[:,0] * EF[:,1] - BC[:,1] * EF[:,0] # (P,)

        a = np.stack((A * EF[:,1][:,None] - D * BC[:,1][:,None], D * BC[:,0][:,None] - A * EF[:,0][:,None]), axis=-1) # (P,T,2)

        pos = a / np.where(div == 0.0, 1e-9, div)[:,None,None] # (P,T,2)

        valid    = div != 0.0                                                          # (P,)
        dists    = np.linalg.norm(pos[None,...] - devParams[:,None,None,0:2], axis=-1) # (N,P,T)
        errs     = np.abs(dists - distances[:,None,:])                                 # (N,P,T)
        totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)               # (P,T)

        bestPairIdx  = np.argmin(totalErr, axis=0)     # (T,)
        timeIdx      = np.arange(bestPairIdx.shape[0]) # (T,)
        bestTotalErr = totalErr[bestPairIdx,timeIdx]   # (T,)
        validMask    = np.isfinite(bestTotalErr)       # (T,) 
        finalPos     = pos[bestPairIdx,timeIdx,:]      # (U,2)
        return (finalPos, validMask)
Positions().Define(TLSE)