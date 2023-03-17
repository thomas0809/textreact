import argparse, random, sys
from rdkit import Chem
from rdkit.Chem import AllChem

def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

class SmilesResolver:
    solvents = {
        'ethers':{'C1COCCO1', 'COCCOC', 'C1CCOC1', 'COC', 'CCOCC', 'COCCOCCOC', 'CCOCCOCC', 'COCOC', 'CC1CCCO1', 'COC(C)(C)C', 'C1COCCO1'},
        'alcohols': {'CO', 'CCO', 'OCCCO', 'CCCO', 'CC(C)O', 'CC(C)(C)O', 'CC(C)CO', 'OCCO', 'CCCCO'},
        'amides':{'CN(C)C=O', 'CC(=O)N(C)C' },
        'aromatics':{'c1ccccc1', 'Cc1ccccc1', 'FC(F)(F)c1ccccc1'},
        'lowbp': {'CC#N', 'CC(C)=O'},
        'other': {'ClCCl', 'ClC(Cl)Cl', 'N#Cc1ccccc1', 'CCOC(C)=O', 'CCCCCCC', 'CCCCCC' },
    }
    bases = { #carbonates, phosphates, fluorides, hydroxides, amines, acetates, and other/miscellaneous.
        'carbonate' : {'O=C([O-])[O-]', 'O=C([O-])O', 'OC([O-])=O'},
        'hydroxide' : {'[OH-]'},
        'phosphate' : {'O=P([O-])([O-])[O-]', 'O=P([O-])([O-])O', 'O=P([O-])(O)O', '[O-][P]([O-])([O-])=O'},
        'acetate' : {'CC(=O)[O-]'},
        'fluoride' : {'[F-]'},
        'other': {'CC(C)(C)[O-]', 'C[O-]', 'O=C([O-])c1cccs1', }, #alcoxide
    }

    top7base = {
        'other' : '[K+].CC(C)(C)[O-]',
        'acetate' : 'CC(=O)[O-].[K+]',
        'amine' : 'CCN(CC)CC',
        'hydroxide' : '[OH-].[Na+]',
        'carbonate' : 'O=C([O-])[O-].[Na+].[Na+]',
        'phosphate' : 'O=P([O-])([O-])[O-].[Na+].[Na+].[Na+]',
        'fluoride' : '[K+].[F-]',
    }

    top6solvent = {
        'alcoArom' : 'CCO.Cc1ccccc1.O',
        'ethers' : 'C1COCCO1',
        'other' : 'CCOC(C)=O.O.C1COCCO1',
        'water/ethers' : 'C1COCCO1.O',
        'polar' : 'CN(C)C=O.O',
        'aromatics' : 'Cc1ccccc1',
    }

    top13solvent = {
        'alcohols/aromatics' : 'Cc1ccccc1.CCO',
        'ethers' : 'C1COCCO1',
        'water/alcohols/aromatics' : 'Cc1ccccc1.O.CCO',
        'water/aromatics' : 'Cc1ccccc1.O',
        'water/ethers' : 'C1COCCO1.O',
        'amides' : 'CN(C)C=O',
        'aromatics' : 'Cc1ccccc1',
        'alcohols' :'CCO',
        'water/alcohols' : 'CCCO.O',
        'water' :'O',
        'water/amides' :'CN(C)C=O.O',
        'water/lowbp' : 'O.CC#N',
        'other' : '',
    }

    def __init__(self, ):
        pass

    def getBaseSolvClassNum(self, base, solv, mode):
        bases = sorted(self.top7base)
        if mode == 'base7solv6':
            solvents = sorted(self.top6solvent)
        elif mode == 'base7solv13':
            solvents = sorted(self.top13solvent)
        else:
            raise NotImplementedError
        return bases.index(base), solvents.index(solv)

    def getBasesAndSolvents(self, mode):
        retDict = {'bases':[], 'baseNames':[], 'solvents':[], 'solventNames':[]}
        for name in sorted(self.top7base):
            retDict['bases'].append(self.top7base[name])
            retDict['baseNames'].append(name)
        if 'solv6' in mode:
            solvDict = self.top6solvent
        elif 'solv13' in mode:
            solvDict = self.top13solvent
        else:
            raise NotImplementedError
        for name in sorted(solvDict):
            retDict['solventNames'].append(name)
            retDict['solvents'].append(solvDict[name] )
        return retDict

    def resolvSolvent(self, smiset):
        detected = set()
        for smi in smiset:
            if type(smi) != str:
                detected.update( self.resolvSolvent(smi))
                continue
            if smi == 'O':
                detected.add('water')
                continue
            for klass in self.solvents:
                if smi in self.solvents[klass]:
                    detected.add(klass)
                    break
            else:
                #raise
                detected.add('other')
        return detected

    def getOldSolventClass(self, smiset):
        #water/ethers, ethers, water/alcohols/aromatics, water/amides, alcohols/aromatics,
        # aromatics, amides, water/aromatics, low boiling polar aprotic solvents /water, water/alcohols
        #, water, alcohols, and other.

        klasses = self.resolvSolvent(smiset)
        if len(klasses) == 1:
            klas = tuple(klasses)[0]
            if klas in {'water', 'ethers', 'aromatics', 'amides', 'alcohols'}:
                return klas
        if len(klasses) == 2:
            if 'water' in klasses and 'amides' in klasses:
                return 'water/amides'
            if 'water' in klasses and 'ethers' in klasses:
                return 'water/ethers'
            if 'water' in klasses and 'alcohols' in klasses:
                return 'water/alcohols'
            if 'water' in klasses and 'lowbp' in klasses:
                return 'water/lowbp'
            if 'water' in klasses and 'aromatics' in klasses:
                return 'water/aromatics'
            if 'alcohols' in klasses and 'aromatics' in klasses:
                return 'alcohols/aromatics'
        if len(klasses) == 3:
            if 'water' in klasses and 'alcohols' in klasses and 'aromatics' in klasses:
                return 'water/alcohols/aromatics'
        return 'other'

    def getNewSolventClass(self,smiset):
        #The more “coarse-grained” classification distinguished six solvent types:
        #polar == {alcohols, polar solvents/water, water/alcohols, water/amides, water, amides}, 
        #alcoArom == {water/aromatics, alcohols/aromatics, water/alcohols/aromatics},
        # {aromatics}, {ethers}, {water/ethers}, {other}
        klasses = self.resolvSolvent(smiset)
        if len(klasses) == 1:
            klas = tuple(klasses)[0]
            if klas in {'aromatics', 'ethers'}:
                return klas
            if klas in {'alcohols', 'amides', 'water'}:
                return 'polar'
        if len(klasses) == 2:
            if 'water' in klasses and 'ethers' in klasses:
                return 'water/ethers'
            if 'water' in klasses and 'alcohols' in klasses:
                return 'polar'
            if 'water' in klasses and 'amides' in klasses:
                return 'polar'
            if 'water' in klasses and 'lowbp' in klasses:
                return 'polar'
            if 'water' in klasses and 'aromatics' in klasses:
                return 'alcoArom'
            if 'alcohols' in klasses and 'aromatics' in klasses:
                return 'alcoArom'
        if len(klasses) == 3:
            if 'water' in klasses and 'aromatics' in klasses and 'alcohols' in klasses:
                return 'alcoArom'
        return 'other'

    def detectBaseClass(self, smiset):
        detected = []
        for basetype in self.bases:
            #print("BA", basetype, type(self.bases[basetype]), type(smiset))
            intersec =  self.bases[basetype].intersection(smiset)
            if intersec:
                detected.append( (basetype, intersec) )
        if len(detected) == 1:
            return detected[0][0]

        if not detected:
            amines = [s for s in smiset if 'N' in s and not 'Na' in s ]
            if amines:
                return 'amine'
        if len(detected) > 1:
            klas = [x[0] for x in detected]
            if 'carbonate' in klas:
                return 'carbonate'
            if 'phosphate' in klas:
                return 'phosphate'
        #print("DET", detected, "FROM", smiset)
        return 'other'


    def resolvBase(self, smisetset):
        detected = set()
        for smiset in smisetset:
            detected.add( self._getBaseType(smiset))
        return detected

    def _getBaseType(self, smiset):
        if isinstance(smiset,tuple) and len(smiset) == 1:
            smiset = smiset[0]
        if isinstance(smiset,str):
            if 'N' in smiset or 'n' in smiset:
                return 'amine'
        else:
            for klass in self.bases:
                for smi in self.bases[klass]:
                    if smi in smiset:
                        return klass
        print("SMISET", smiset)
        raise NotImplementedError


def parseArgs():
    parser = argparse.ArgumentParser(description='train yieldBERT')
    parser.add_argument('--parsed', type=str, nargs='+', required=True, help='file with parsed reactions')
    parser.add_argument('--auxoutformat', choices=['none', 'base7solv6', 'base7solv13'], default='none',
         help='auxilary input file with additional fake reaction with base/solvent from each class. Fake reactions are directly below the true one')
    parser.add_argument('--yieldforfake', default=-1, type=float, help='for fake reactions set this yield')
    parser.add_argument('--outprefix', type=str, required=True, help='directory wihere results will be stored')
    parser.add_argument('--includewaste', action='store_true', help='if selected also smiles with unknown role will be included')
    parser.add_argument('--makecanonsmiles', action='store_true', help='make substrate and product canonical')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--cv', type=int, default=1, help='create N folds if N < 1 all data in one id not CV')
    parser.add_argument('--conditionsasreactants', action='store_true',
        help='if selected conditions will be added to starting material sbs.cond>>prod instead of sbs>cond>prod')
    parser.add_argument('--outputformat', choices=['rx', 'rxrecom', 'GAT', 'fp512r3', 'BertCondition'], default='rx', help='output format either full rx or csv for relGAT')
    args = parser.parse_args()
    return args

def makeCatDict(rxlist, args):
    pdligDict = dict()
    pdligNum = 0
    usedNums = list()
    baseDict = dict()
    solvDict = dict()
    for rxline in rxlist:
        pdlig = frozenset(set(rxline['catalysts']).union(set(rxline['ligands'])))
        #pdlig = frozenset(set().union(set(rxline['ligands'])))
        if not pdlig in pdligDict:
            pdligDict[pdlig] = pdligNum
            usedNums.append(pdligNum)
            pdligNum += 1
    return pdligDict, usedNums

def buildRxSmiles(dane, includeWaste):
    #print("DANE", dane)
    condlist = _makeSmiList(dane['catalysts']) + _makeSmiList(dane['ligands'])
    if includeWaste and 'other' in dane:
        condlist += _makeSmiList(dane['other'])
    return condlist, _makeSmiList(dane['bases']), _makeSmiList(dane['solvents']), dane['yield']


def _makeSmiList(data):
    condlist = []
    for smi in data:
        if isinstance(smi, str):
            condlist.append(smi)
        elif isinstance(smi,(tuple,list,set)):
            condlist.extend(smi)
        else:
            raise NotImplementedError
    return condlist

def parseLineToGAT(line, smiSolver, args, pdligName):
    solv = frozenset(line['solvents'])
    bs = frozenset(line['bases'])
    #smiSolver.resolvSolvent(solv)
    basecl =  smiSolver.resolvBase(bs)
    if len(basecl) == 1:
        baseClass = tuple(basecl)[0]
    else:
        baseClass = 'other'
    if args.auxoutformat == 'base7solv13':
        solvClass = smiSolver.getOldSolventClass(solv)
        if solvClass == 'other':
            print("SOLV", solv, "<= other")
    else:
        solvClass = smiSolver.getNewSolventClass(solv)
    sbs = [line['rx'][0][0], line['rx'][0][1], ]
    prod = line['rx'][0][2]
    #Reactant1;Reactant2;Product;Yield;temperature;M;L;B;S
    #C[...]c(B(O)O);C[...](Br);CC1[...]C(C)C=C2)C=C1;13.0;80;M0M;M37M;M51M;M60M
    if args.makecanonsmiles:
        sbsmols = [Chem.MolFromSmiles(s) for s in sbs]
        sbs = []
        for m in sbsmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            sbs.append(Chem.MolToSmiles(m))
        prodmols = [Chem.MolFromSmiles(s) for s in prod]
        prod = []
        for m in prodmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            prod.append(Chem.MolToSmiles(m))
    pdlig =  frozenset(set(line['catalysts']).union(set(line['ligands'])))
    ML = 'M'+str(pdligName[pdlig]%10)+'M'
    #ML = 'M0M'
    numML = 10
    B, S = smiSolver.getBaseSolvClassNum(baseClass, solvClass, args.auxoutformat)
    B = 'B'+str(numML+B)+'B'
    S = 'S'+str(numML+7+S)+'S'
    fakeT = '20'
    #B = baseClass
    #S = solvClass
    return sbs[0], sbs[1], prod[0], line['yield'], fakeT, ML, B, S

def parseLineToBertCondition(line, smiSolver, args, pdligName):
    
    def convert_x(x):
        if isinstance(x, tuple):
            return '.'.join(x)
        else:
            return x
            
    
    solv = frozenset(line['solvents'])
    solv_smiles_list = sorted([canonicalize_smiles(convert_x(x)) for x in list(solv)])
    solvents = '[SPLIT]'.join(solv_smiles_list)
    bs = frozenset(line['bases'])
    bs_smiles_list = sorted([canonicalize_smiles(convert_x(x)) for x in list(bs)])
    bases = canonicalize_smiles('.'.join(bs_smiles_list))
    #smiSolver.resolvSolvent(solv)
    basecl =  smiSolver.resolvBase(bs)
    if len(basecl) == 1:
        baseClass = tuple(basecl)[0]
    else:
        baseClass = 'other'
    if args.auxoutformat == 'base7solv13':
        solvClass = smiSolver.getOldSolventClass(solv)
        if solvClass == 'other':
            print("SOLV", solv, "<= other")
    else:
        solvClass = smiSolver.getNewSolventClass(solv)
    sbs = [line['rx'][0][0], line['rx'][0][1], ]
    prod = line['rx'][0][2]
    #Reactant1;Reactant2;Product;Yield;temperature;M;L;B;S
    #C[...]c(B(O)O);C[...](Br);CC1[...]C(C)C=C2)C=C1;13.0;80;M0M;M37M;M51M;M60M
    if args.makecanonsmiles:
        sbsmols = [Chem.MolFromSmiles(s) for s in sbs]
        sbs = []
        for m in sbsmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            sbs.append(Chem.MolToSmiles(m))
        prodmols = [Chem.MolFromSmiles(s) for s in prod]
        prod = []
        for m in prodmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            prod.append(Chem.MolToSmiles(m))
    catalysts_smiles_list = sorted([canonicalize_smiles(convert_x(x)) for x in list(line['catalysts'])])
    catalysts = canonicalize_smiles('.'.join(catalysts_smiles_list))
    ligands_smiles_list = sorted([canonicalize_smiles(convert_x(x)) for x in list(line['ligands'])])
    ligands = canonicalize_smiles('.'.join(ligands_smiles_list))
    pdlig =  frozenset(set(line['catalysts']).union(set(line['ligands'])))
    ML = 'M'+str(pdligName[pdlig]%10)+'M'
    #ML = 'M0M'
    numML = 10
    B, S = smiSolver.getBaseSolvClassNum(baseClass, solvClass, args.auxoutformat)
    B = 'B'+str(numML+B)+'B'
    S = 'S'+str(numML+7+S)+'S'
    fakeT = '20'
    #B = baseClass
    #S = solvClass
    return sbs[0], sbs[1], prod[0], line['yield'], fakeT, ML, B, S, catalysts, ligands, bases, solvents

def parseLineToRxRecomp(line, smiSolver, args):
    sbs = [line['rx'][0][0], line['rx'][0][1], ]
    prod = line['rx'][0][2]

    if args.makecanonsmiles:
        sbsmols = [Chem.MolFromSmiles(s) for s in sbs]
        sbs = []
        for m in sbsmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            sbs.append(Chem.MolToSmiles(m))
        prodmols = [Chem.MolFromSmiles(s) for s in prod]
        prod = []
        for m in prodmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            prod.append(Chem.MolToSmiles(m))
    rxn = '.'.join(sbs) +'>>'+ '.'.join(prod)

    solv = frozenset(line['solvents'])
    bs = frozenset(line['bases'])
    #smiSolver.resolvSolvent(solv)
    basecl =  smiSolver.resolvBase(bs)
    if len(basecl) == 1:
        baseClass = tuple(basecl)[0]
    else:
        baseClass = 'other'
    if args.auxoutformat == 'base7solv13':
        solvClass = smiSolver.getOldSolventClass(solv)
    else:
        solvClass = smiSolver.getNewSolventClass(solv)

    return rxn, line['yield'], baseClass, solvClass

def parseLineToFP(line,smiSolver, args, fplen=512, radius=3):
    solv = frozenset(line['solvents'])
    bs = frozenset(line['bases'])
    basecl =  smiSolver.resolvBase(bs)
    if len(basecl) == 1:
        baseClass = tuple(basecl)[0]
    else:
        baseClass = 'other'
    if args.auxoutformat == 'base7solv13':
        solvClass = smiSolver.getOldSolventClass(solv)
    else:
        solvClass = smiSolver.getNewSolventClass(solv)
    #
    B, S = smiSolver.getBaseSolvClassNum(baseClass, solvClass, args.auxoutformat)


    sbs = [line['rx'][0][0], line['rx'][0][1], ]

    if args.makecanonsmiles:
        sbsmols = [Chem.MolFromSmiles(s) for s in sbs]
        sbs = []
        for m in sbsmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            sbs.append(Chem.MolToSmiles(m))

    vector = [B, S]
    fp1 = AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sbs[0]),radius, nBits=fplen ).ToBitString()
    _ = [ vector.append(x) for x in fp1]
    fp2 = AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sbs[1]),radius, nBits=fplen ).ToBitString()
    _ = [vector.append(x) for x in fp2]
    vector.append( line['yield'])
    return vector


def parseLineToRx(line, smiSolver, args):
    solv = frozenset(line['solvents'])
    bs = frozenset(line['bases'])
    basecl =  smiSolver.resolvBase(bs)
    if len(basecl) == 1:
        baseClass = tuple(basecl)[0]
    else:
        baseClass = 'other'
    if args.auxoutformat == 'base7solv13':
        solvClass = smiSolver.getOldSolventClass(solv)
    else:
        solvClass = smiSolver.getNewSolventClass(solv)
    rxes = []
    sbs = [line['rx'][0][0], line['rx'][0][1], ]
    prod = line['rx'][0][2]

    if args.makecanonsmiles:
        sbsmols = [Chem.MolFromSmiles(s) for s in sbs]
        sbs = []
        for m in sbsmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            sbs.append(Chem.MolToSmiles(m))
        prodmols = [Chem.MolFromSmiles(s) for s in prod]
        prod = []
        for m in prodmols:
            _ = [ a.SetAtomMapNum(0) for a in m.GetAtoms()]
            prod.append(Chem.MolToSmiles(m))

    cond, base, solvent, yld = buildRxSmiles(line, args.includewaste)
    if  args.includewaste:
        if 'notUsedSbs' in line:
            sbs.extend( line['notUsedSbs'])
        if 'notUsedProd' in line:
            prod.extend( line['notUsedProd'])
        if 'other' in line:
            cond.extend( line['other'])
    fullcond = cond+base+solvent
    if args.conditionsasreactants:
        rx = '.'.join(sbs)+'.'+'.'.join(fullcond)+'>>'+'.'.join(prod)
    else:
        rx = '.'.join(sbs)+'>'+'.'.join(fullcond)+'>'+'.'.join(prod)
    rxes.append(rx)
    if args.auxoutformat != 'none':
        basolv = smiSolver.getBasesAndSolvents(args.auxoutformat)
        print("XX", basolv, baseClass, solvClass)
        ##same as true reaction
        baseSameClass = basolv['bases'][ basolv['baseNames'].index(baseClass) ]
        solvSameClass = basolv['solvents'][ basolv['solventNames'].index(solvClass) ]
        fullcond = cond + [baseSameClass, solvSameClass]
        ## other group
        for base in basolv['bases']:
            #print("BAS", base)
            for solv in basolv['solvents']:
                fullcond = cond+ [base, solv]
                if base == baseSameClass and solv == solvSameClass:
                    continue
                if args.conditionsasreactants:
                    rx = '.'.join(sbs)+'.'+'.'.join(fullcond)+'>>'+'.'.join(prod)
                else:
                    rx = '.'.join(sbs)+'>'+'.'.join(fullcond)+'>'+'.'.join(prod)
                rxes.append(rx)
        print(len(rxes))
    return rxes, yld


def printLines(lines, smiSolver, args):
    pdligDict, pdligUsedName = makeCatDict(lines, args)
    if args.cv > 1:
        foldlen = len(lines) // args.cv
        foldidx = [( x*foldlen, (x+1)*foldlen) for x in range(args.cv)]
        foldidx[-1] = (foldidx[-1][0], len(lines))
    else:
        foldidx = [(0, len(lines)), ]
    for idx, fold in enumerate(foldidx):
        if not args.outputformat == 'BertCondition':
            fhmain = open(f'{args.outprefix}_train{idx}.txt', 'w')
            fhvalidation = open(f'{args.outprefix}_valid{idx}.txt', 'w')
            fhvalidext = open(f'{args.outprefix}_validext{idx}.txt', 'w')
        else:
            fhmain = open(f'{args.outprefix}_dataset.csv', 'w')
        fakeT='temperature;'
        if args.outputformat == 'GAT':
            print("Reactant1;Reactant2;Product;Yield;"+fakeT+"ML;B;S", file=fhmain)
            print("Reactant1;Reactant2;Product;Yield;"+fakeT+"ML;B;S", file=fhvalidation)
            print("Reactant1;Reactant2;Product;Yield;"+fakeT+"ML;B;S", file=fhvalidext)
        elif args.outputformat == 'BertCondition':
            print("Reactant1;Reactant2;Product;Yield;"+fakeT+"ML;B;S;catalysts;ligands;bases;solvents",file=fhmain)
        for lid, line in enumerate(lines):
            if not args.outputformat == 'BertCondition':
                if lid >= fold[0] and lid < fold[1]:
                    outfile = fhvalidation
                    isValidation = True
                else:
                    outfile = fhmain
                    isValidation = False
            else:
                outfile = fhmain
                isValidation = False

            if args.outputformat == 'GAT':
                print( *parseLineToGAT(line, smiSolver, args, pdligDict), sep=';', file=outfile)
            elif args.outputformat == 'rxrecom':
                rxinfo = parseLineToRxRecomp(line, smiSolver, args)
                print(*rxinfo, sep=';', file=outfile)
            elif args.outputformat == 'fp512r3':
                rxinfo = parseLineToFP(line, smiSolver, args, fplen=512, radius=3)
                print(*rxinfo, file=outfile)
            elif args.outputformat == 'rx':
                rxes, yld = parseLineToRx(line, smiSolver, args)
                print(rxes[0], yld, file=outfile)
                if isValidation:
                    print(rxes[0], yld, file=fhvalidext)
                    for rx in rxes:
                        print(rx, args.yieldforfake, file=fhvalidext)
            elif args.outputformat == 'BertCondition':
                print( *parseLineToBertCondition(line, smiSolver, args, pdligDict), sep=';', file=outfile)
            else:
                raise NotImplementedError
        if not args.outputformat == 'BertCondition':
            fhmain.close()
            fhvalidation.close()
            fhvalidext.close()
        else:
            fhmain.close()


if __name__ == "__main__":
    args = parseArgs()
    smiSolver = SmilesResolver()
    lines = []
    for fn in args.parsed:
        for line in open(fn):
            if not 'rx' in line:
                continue
            lines.append(eval(line))
    if args.shuffle:
        random.shuffle(lines)
    printLines(lines, smiSolver, args)


