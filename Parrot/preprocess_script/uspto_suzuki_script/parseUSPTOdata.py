import os
import re, statistics
from rdkit import Chem
from rdkit.Chem import AllChem

#the script is intended to extract from
# 1976_Sep2016_USPTOgrants_smiles.rsmi and
#reactions which have yield given and 'Pd' as well as 'B' in reaction smiles
#suzukiRx = AllChem.ReactionFromSmarts('[#6:3][BX3:4]([OX2:5])[OX2:6].[#6:1][Br,I,Cl:2]>>[#6:1][#6:3].[*:2].[B:4]([O:5])[O:6]')
suzukiRx = AllChem.ReactionFromSmarts('[c:3][BX3:4]([OX2H:5])[OX2H:6].[c:1][Br,I:2]>>[c:1][c:3].[*:2].[B:4]([O:5])[O:6]')
het1 = Chem.MolFromSmarts('[cr5][Br,I,BX3]')
het2 = Chem.MolFromSmarts('[n][cr6][Br,I,BX3]')
het3 = Chem.MolFromSmarts('[n][a][cr6][Br,I,BX3]')
het4 = Chem.MolFromSmarts('[n][a][a][cr6][Br,I,BX3]')



DEBUG_LIGAND = False
DEBUG_PD = False

specialCasesConditions = {
    ('C1CCOC1', 'C=C[CH2-]', 'C=C[CH2-]', 'Cl[Pd+]', 'Cl[Pd+]', 'O') : {'solvents':('C1CCOC1', 'O'), 'Pd':[('C=C[CH2-]', 'Cl[Pd+]')]},
    ('C1COCCO1', 'O=C([O-])[O-]', '[Na+]', '[Na+]', '[Cl-]', '[Cl-]', 'c1ccc(P(c2ccccc2)c2ccccc2)cc1', 'c1ccc(P(c2ccccc2)c2ccccc2)cc1', '[Pd+2]') : {'solvents': ['C1COCCO1',],
        'bases': [('[Na+]', '[Cl-]'),], 'ligands': ('c1ccc(P(c2ccccc2)c2ccccc2)cc1',), 'Pd': [('O=C([O-])[O-]', '[Pd+2]'),] },
    ('CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', 'CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', '[Cl-]', '[Cl-]', '[Fe+2]', '[Pd+2]', 'O') :{'solvents':['O',],
        'Pd':[('CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', '[Pd+2]', 'CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C'),]},
    ('CC(=O)[O-]', '[Pd+2]', '[Pd+2]', 'CC(=O)[O-]', 'CC(=O)[O-]', 'CC(=O)[O-]', 'O', 'CN(C)C=O', 'COCCOC') : {'solvents':['O', 'CN(C)C=O', 'COCCOC'],
        'Pd':[('CC(=O)[O-]', '[Pd+2]', 'CC(=O)[O-]'),] },
    ('COCCOC', 'O', 'CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', 'CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', '[Cl-]', '[Cl-]', '[Fe+2]', '[Pd+2]') : {'solvents':['COCCOC', 'O'],
        'Pd':[ ('CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', 'CC(C)(C)P(c1ccc[cH-]1)C(C)(C)C', '[Pd+2]'),],  },
    ('CS(C)=O', 'CO', 'C1COCCO1', 'C=C[CH2-]', 'C=C[CH2-]', 'Cl[Pd+]', 'Cl[Pd+]', 'F[B-](F)(F)F', 'CC(C)(C)[PH+](C(C)(C)C)C(C)(C)C', 'CCOC(C)=O') : { 'Pd':[('C=C[CH2-]', 'Cl[Pd+]'),],
        'ligands': ['F[B-](F)(F)F', 'CC(C)(C)[PH+](C(C)(C)C)C(C)(C)C',], 'solvents':['CS(C)=O', 'CO', 'C1COCCO1', 'CCOC(C)=O'] },
    ('C1COCCO1', 'O', '[Cl-]', '[Cl-]', '[Cl-]', '[Cl-]', '[Na]', '[Na]', '[Pd+4]', 'CC(C)(C)P(CCCS(=O)(=O)O)C(C)(C)C') : {'solvents':['C1COCCO1', 'O',],
        'Pd':[ ('[Cl-]', '[Cl-]', '[Cl-]', '[Cl-]', '[Na+]', '[Na+]', '[Pd+4]')], 'ligands':('CC(C)(C)P(CCCS(=O)(=O)O)C(C)(C)C',) },
    ('C1CCOC1', 'CC(C)c1cc(C(C)C)c(-c2cccc(P(C3CCCCC3)C3CCCCC3)c2)c(C(C)C)c1', 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1', 'Cl[Pd+]', 'Cl[Pd+]',
        'NCCc1[c-]cccc1', 'Nc1ccccc1-c1[c-]cccc1'): {'solvents':['C1CCOC1',], 'ligands':['CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1',
        'CC(C)c1cc(C(C)C)c(-c2cccc(P(C3CCCCC3)C3CCCCC3)c2)c(C(C)C)c1'], 'Pd':[('Cl[Pd+]', 'NCCc1[c-]cccc1'), ('Cl[Pd+]', 'Nc1ccccc1-c1[c-]cccc1') ] },
    ('CC(=O)[O-]','CC(=O)[O-]','[Pd+2]','c1ccc(-c2ccccc2)cc1'): {'ligands':['CC(C)(C)P(c1ccccc1-c1ccccc1)C(C)(C)C',], 'Pd':[('CC(=O)[O-]', '[Pd+2]', 'CC(=O)[O-]'),] },
    ('CS(C)=O', 'CC(O)=O', 'CC(O)=O', 'C1CCC([N-]C2CCCCC2)CC1', 'C1CCC([N-]C2CCCCC2)CC1', '[Pd+2]') : {'solvents':('CS(C)=O',), 'ligands':('C1CCC(NC2CCCCC2)CC1',), 
        'Pd':[('CC(=O)[O-]', '[Pd+2]', 'CC(=O)[O-]'),] },


}


class ReplaceCmd:
    baseSets = {
        frozenset({'[Na+]', 'O=C([O-])O'}) : ('O=C([O-])O', '[Na+]', '[Na+]'),
        frozenset({'[Na+]', 'O=C([O-])[O-]'}) : ('[Na+]', 'O=C([O-])[O-]'),
        frozenset({'[K+]', 'O=C([O-])O'}) : ('O=C([O-])O', '[K+]', '[K+]'),
        frozenset({'[K+]', 'O=C([O-])[O-]'}) : ('[K+]', 'O=C([O-])[O-]'),
    }
    wrongBases = {
        'CC(C)(C)O[Na]' : ('CC(C)(C)O', '[Na+]'),
        'CC(C)(C)O[K]' : ('CC(C)(C)[O-]', '[K+]'),
        'CC(=O)O[K]' : ('CC(=O)[O-]', '[K+]',),
        'CC(=O)O[Na]' : ('CC(=O)[O-]', '[Na+]'),
        '[Li]O' : ('[Li+]','[OH-]'),
        'C1CCCC(N2CCCCCCN2)CCC1': ('C1CN2CCN1CC', ),
    }
    wrongSolvents = {
        'COC=COC' : ('COCCOC', ),
        'CCOC(C)OCC' : ('CCOCCOCC', ),
        'Cc1ccccc1CCO' : ('Cc1ccccc1', 'CCO'),
        'Cc1ccccc1CO' :  ('Cc1ccccc1', 'CO'),
        'N#CCC1COCCO1' : ('N#CC', 'C1COCCO1'),
    }
    def __init__(self, replaceFile, solventFile, baseFile, ligandFile):
        fh = open(replaceFile)
        self.singleWrongToFull = dict()
        self.replaceWrong = dict()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            #print("LINE", line)
            wrong, correct = line.split('\t')
            wrong = wrong.split('.')
            noNumWrong = tuple([self._removeNumbering(smi) for smi in wrong])
            for smid, smi in enumerate(wrong):
                nonumSmi = noNumWrong[smid]
                self.singleWrongToFull[smi] = set(noNumWrong)
                self.singleWrongToFull[nonumSmi] = set(noNumWrong)
            self.replaceWrong[ frozenset(noNumWrong)] = correct
        self.solvent, self.solventComplex = self.parseCondFile(solventFile)
        self.base, self.baseComplex = self.parseCondFile(baseFile)
        self.ligand, self.ligandComplex = self.parseCondFile(ligandFile)
        #self.simpleBase = {'[O-]', '[K+]', '[Na+]', 
        #print("BASES", self.base)
        #print("COMPLEX", self.baseComplex)

    @staticmethod
    def parseCondFile(fn):
        retDict = dict()
        retComplexDict = dict()
        fh = open(fn)
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split('\t')
            smi, name = elems[0:2]
            canonsmi = smi
            if len(elems) ==3 and elems[2]:
                canonsmi = elems[2]
            if '.' in smi:
                retComplexDict[ frozenset(smi.split('.'))] = {'raw':tuple(smi.split('.')), 'canon':tuple(canonsmi.split('.'))}
            else:
                retDict[smi] = canonsmi
        fh.close()
        return retDict, retComplexDict

    def isMisspelledBase(self, smi):
        if smi in self.wrongBases:
            return self.wrongBases[smi]
        return False

    def isMisspelledSolvent(self, smi):
        if smi in self.wrongSolvents:
            return self.wrongSolvents[smi]
        return False

    def isBase(self, smi):
        if smi in self.base:
            return self.base[smi]
        return False

    def isLigand(self, smi):
        if smi in self.ligand:
            return self.ligand[smi]
        return False

    @staticmethod
    def _isStoichCorrect(cmdPartsList, fullList):
        cmdPartsCount = {cmdPart:cmdPartsList.count(cmdPart) for cmdPart in set(cmdPartsList)}
        for partOfCmd in cmdPartsCount:
            if fullList.count(partOfCmd) < cmdPartsCount[partOfCmd]:
                return False
        return True

    def getComplexLigand(self, smiList, minlen=3):
        #for ferrocene-based ligand mostly
        smiSet = set(smiList)
        torm = []
        ligands = []
        for cligand in self.ligandComplex:
            if len(cligand) < minlen:
                if DEBUG_LIGAND:
                    print("LIGAND TO SHORT", cligand, "LIMIT", minlen)
                continue
            if DEBUG_LIGAND:
                print("LIGAND", cligand," in", smiList, ":::", cligand.issubset(smiSet), self._isStoichCorrect( self.ligandComplex[cligand]['raw'], smiList) )
            if cligand.issubset(smiSet) and self._isStoichCorrect( self.ligandComplex[cligand]['raw'], smiList ):
                torm.extend( self.ligandComplex[cligand]['raw'])
                ligands.append( self.ligandComplex[cligand]['canon'])
        return ligands, torm

    def getIonicBase(self, smilesSet, debug=False):
        keys = list()
        values = list()
        if smilesSet:
            for base in self.baseComplex:
                if debug:
                    print("BASE", base, base.issubset(smilesSet), "IN", smilesSet)
                if base.issubset(smilesSet):
                    keys.append(base)
                    values.append(self.baseComplex[base]['canon'])
        return values, keys

    def getBaseSet(self, smilesSet):
        keys = list()
        values = list()
        for bset in self.baseSets:
            if bset.issubset(smilesSet):
                keys.append(bset)
                values.append( self.baseSets[bset])
        return values, keys

    def isSolvent(self,smi):
        if smi in self.solvent:
            return True
        return False

    @staticmethod
    def _removeNumbering(smi):
        return re.sub( ':\d+', '', smi)

    def getNormSmiles(self, smiles):
        smiles = smiles.split('.')
        orygWrong = set()
        noNumWrong = set()
        okSmiles = []
        for smi in smiles:
            noNumSmi = self._removeNumbering(smi)
            if smi in self.singleWrongToFull or noNumSmi in self.singleWrongToFull:
                orygWrong.add(smi)
                noNumWrong.add(noNumSmi)
            else:
                try:
                    canonSmi = Chem.CanonSmiles(smi)
                except:
                    print("CANNOT CANON", smi, "FROM", smiles)
                    raise
                okSmiles.append(canonSmi)
        matched = set()
        if noNumWrong:
            for singleWrong in self.singleWrongToFull:
                fullWrong = self.singleWrongToFull[singleWrong]
                if fullWrong.issubset(noNumWrong):
                    matched.add(frozenset(fullWrong))
        if len(matched) > 1:
            print("MATCHES", matched)
            raise
        if matched:
            matched = tuple(matched)[0]
            okSmiles.extend(self.replaceWrong[matched])
        return '.'.join(okSmiles)



def parseRxFile(fn, seenSmiles, replace, sep='\t', onlyWithYield=True, selectedRx=('suzuki',), columnToRead={'smiles':(0,), 'yields':(5,4)},
                excludeFile='uspto_exclude_list.smi'):
    #ReactionSmiles  PatentNumber    ParagraphNum    Year    TextMinedYield  CalculatedYield
    fh = open(fn)
    excludeRx = set()
    if excludeFile:
        with open(excludeFile) as exl:
            for line in exl:
                line = line.strip()
                if line:
                    excludeRx.add(line)
    header = []
    retList = []
    wrongDict = dict()
    issuzuki = 0
    notsuzuki = 0
    notUsedSbs = set()
    notUsedProd = set()
    other =set()
    for lid, line in enumerate(fh):
        line = line[:-1]
        if lid == 0:
            header = line.split(sep)
            continue
        line = line.split(sep)
        rawSmiles = [ getRxSmiles(line[x]) for x in columnToRead['smiles']]
        rawYields = [line[x] for x in columnToRead['yields']]
        yields = getYield(rawYields)
        if not yields:
            continue
        if len(rawSmiles) != 1:
            raise NotImplementedError
        rawSmiles = rawSmiles[0].replace('[Fe]', '[Fe+2]')
        if rawSmiles in excludeRx:
            continue
        if not isSuzukiFastCheck(rawSmiles):
            continue
        if rawSmiles in seenSmiles:
            continue
        seenSmiles.add(rawSmiles)
        suzukiInfo = getSuzukiRoles(rawSmiles, replace)
        if not 'rx' in suzukiInfo:
            print("SPEC", suzukiInfo)
            continue
        if not suzukiInfo['rx']:
            notsuzuki += 1
            continue
        suzukiInfo['yield'] = yields
        suzukiInfo = addMissedConditionsFromSubstrate(replace, suzukiInfo)
        suzukiInfo = makeCleanUp(suzukiInfo)
        if 'other' in suzukiInfo:
            other.update( suzukiInfo['other'])
        print("SUZIKIINFO", line, "IS::",  suzukiInfo)
        if 'notUsedSbs' in suzukiInfo or 'notUsedProd' in suzukiInfo:
            print("FFFFFF", rawSmiles, '.'.join(suzukiInfo.get('notUsedSbs',[])), '.'.join(suzukiInfo.get('notUsedProd',[])))
        if 'other' in suzukiInfo:
            print("OOOOO", rawSmiles, '.'.join(suzukiInfo['other']))
        if 'notUsedSbs' in suzukiInfo:
            notUsedSbs.update( set(suzukiInfo['notUsedSbs']))
        if 'notUsedProd' in suzukiInfo:
            notUsedProd.update(set(suzukiInfo['notUsedProd']))
        issuzuki +=1
        retList.append({'raw':line, 'parsed':suzukiInfo})
    fh.close()
    for i in wrongDict:
        print("WRONG::", i, wrongDict[i])
    print("NOT S", notsuzuki, "IS SUZUKI", issuzuki)
    print("OTHER", makeCanonicalSmilesSet(other))
    print("NOT USED SBS", makeCanonicalSmilesSet(notUsedSbs))
    print("NOT USED PRD", makeCanonicalSmilesSet(notUsedProd))
    return retList, seenSmiles


def makeCleanUp(suzukiInfo):
    if 'other' in suzukiInfo and not suzukiInfo['bases']:
        if len(suzukiInfo['other']) == 1 and suzukiInfo['other'] in {'CC(=O)O'}:
            _ = suzukiInfo.pop('other')
            suzukiInfo['bases'].add( ('CC(=O)[O-]', '[Na+]'))
        elif len(suzukiInfo['other']) == 1 and suzukiInfo['other'] in {'O=P(O)(O)O'}:
            _ = suzukiInfo.pop('other')
            suzukiInfo['bases'].add( ('O=P([O-])([O-])[O-]', '[K+]', '[K+]', '[K+]'))
    return suzukiInfo

def _removeAtomMapNum(smi):
    mol = Chem.MolFromSmiles(smi)
    _ = [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def addMissedConditionsFromSubstrate(replaceObj, infoDict, debug=False, removeAtomMapNum=True):
    rmOther = []
    if 'notUsedSbs' not in infoDict:
        return infoDict
    ##remove solvents
    for poz, smi in enumerate(infoDict.get('notUsedSbs',[])):
        if ':' in smi and removeAtomMapNum:
            nonumsmi  = _removeAtomMapNum(smi)
            infoDict['notUsedSbs'][poz] = nonumsmi
            smi = nonumsmi
        print("CHECKSOLVENT", smi)
        newsolv = replaceObj.isMisspelledSolvent(smi)
        newbase = replaceObj.isMisspelledBase(smi)
        if newsolv:
            infoDict['solvents'].add(newsolv)
            rmOther.append(smi)
        elif newbase:
            infoDict['bases'].add(newbase)
            rmOther.append(smi)
        elif replaceObj.isSolvent(smi):
            infoDict['solvents'].add(smi)
            rmOther.append(smi)
            print("ISSOLVENT", smi)
    if debug:
        print("BEFORE", infoDict['notUsedSbs'])
    if rmOther:
        _ = [infoDict['notUsedSbs'].remove(smi) for smi in rmOther]
    ## remove bases
    rmOther = []
    for smi in infoDict.get('notUsedSbs',[]):
        if replaceObj.isBase(smi):
            infoDict['bases'].add(smi)
            rmOther.append(smi)
        elif replaceObj.isLigand(smi):
            if DEBUG_LIGAND:
                print("LIGAND FOUND IN SBS", smi)
            infoDict['ligands'].add(smi)
            rmOther.append(smi)
    if rmOther:
        _ = [infoDict['notUsedSbs'].remove(smi) for smi in rmOther]

    if not infoDict['notUsedSbs']:
        _ = infoDict.pop('notUsedSbs')
        return infoDict
    infoDict['notUsedSbs'] = set(infoDict['notUsedSbs'])
    if rmOther:
        print("RREMOBED", rmOther)
    bases, torm = replaceObj.getIonicBase( infoDict['notUsedSbs'])
    if bases:
        debg = True
    if debug:
        print("IIIIII", bases, torm)
    _ = [ infoDict['bases'].add(b) for b in bases]
    for rmset in torm:
        infoDict['notUsedSbs'] = infoDict['notUsedSbs'] - rmset
    if debug:
        print("AFTER", infoDict['notUsedSbs'], "::", rmOther, "TORN", torm)
    if not infoDict['notUsedSbs']:
        _ = infoDict.pop('notUsedSbs')
    return infoDict


def getRxSmiles(smi):
    smi = [s for s in  smi.split() if '>' in s]
    if len(smi) > 1:
        raise NotImplementedError
    elif len(smi) == 1:
        return smi[0]
    return ''

def makeCanonicalSmilesSet(smiSet):
    newset = set()
    wrongSmi = set()
    for smi in smiSet:
        try:
            mol = Chem.MolFromSmiles(smi)
            for atm in mol.GetAtoms():
                atm.SetAtomMapNum(0)
            newsmi = Chem.MolToSmiles(mol)
            if newsmi:
                newset.add(newsmi)
            else:
                wrongSmi.add(smi)
        except:
            wrongSmi.add(smi)
    return newset, wrongSmi

def isSuzukiFastCheck(smiles):
    if not 'Pd' in smiles:
        return False
    br = smiles.count('Br')
    i = smiles.count('I')
    b = smiles.count('B')
    if not b > br:
        return False
    if br + i < 1:
        return False
    return True

def getSuzukiRoles(smiles, replaceObj):
    sbs, cat, prod = smiles.split('>')
    #replaceObj = 

    sbsSmi = replaceObj.getNormSmiles(sbs)
    catSmi = replaceObj.getNormSmiles(cat).split('.')
    cat = tuple(catSmi)
    if cat in specialCasesConditions:
        return specialCasesConditions[cat]
    sbsSet = set(sbsSmi.split('.'))
    prodSet = set(prod.split('.'))
    rxcomponents, notUsedSbs, notUsedProd  = getSuzukiCompounds(sbsSet, prodSet)
    catSet = set()
    baseSet = set()
    solventSet = set()
    ligandSet = set()
    other = set()

    #lets first get ferrocene-based ligand
    minlen = 3
    for i in range(5): #in some reaction ligands are repeated so removed it several time
        ligands, torm = replaceObj.getComplexLigand(catSmi, minlen=minlen)
        for i in torm:
            catSmi.remove(i)
        for l in ligands:
            if DEBUG_LIGAND:
                print("LIGAND FOUND IN SBS", l, "OF", ligands)
            ligandSet.add(l)
        if not torm:
            if minlen == 3:
                minlen -= 1
            else:
                break
    print("LIGAND SET", ligandSet)
    nocatSmi, catSet = getPdCat(catSmi)
    #print("CATSET", catSet)
    if -1 == catSet:
        print("STRANGE", smiles)
        raise
    for smi in nocatSmi:
        newsolv = replaceObj.isMisspelledSolvent(smi)
        newbase = replaceObj.isMisspelledBase(smi)
        if newsolv:
            solventSet.add(newsolv)
        elif newbase:
            baseSet.add(newbase)
        elif 'P' in smi and ('c' in smi or 'C' in smi):
            ligandSet.add(smi)
        elif replaceObj.isLigand(smi):
            ligandSet.add(smi)
        elif replaceObj.isSolvent(smi):
            solventSet.add(smi)
        elif replaceObj.isBase(smi):
            baseSet.add(smi)
        else:
            other.add(smi)
    bases, removed = replaceObj.getIonicBase(other)
    _ = [baseSet.add(b) for b in bases]
    for r in removed:
        other = other - r
        #elif #solvent #base
    #prodSmi = replaceObj.getNormSmiles(prod)
    retDict = {'rx':rxcomponents, 'catalysts':catSet, 'bases':baseSet, 'solvents':solventSet, 'ligands':ligandSet}
    if other:
        retDict['other'] = other
    if notUsedSbs:
        retDict['notUsedSbs'] = notUsedSbs
    if notUsedProd:
        retDict['notUsedProd'] = notUsedProd
    return retDict


def getSuzukiCompounds(sbsset, prodSet):
    candidateB = [s for s in sbsset if s.count('B') > s.count('Br') and s != 'F[B-](F)(F)F']
    candidateX = [s for s in sbsset if 'Br' in s or 'I' in s]
    #ontherIn = [s for s in sbsset if  s not in candidateB and s not in candidateX]

    molsB = [Chem.MolFromSmiles(s) for s in candidateB]
    molsX = [Chem.MolFromSmiles(s) for s in candidateX]
    #prodSmis = tuple(prodSet)
    prodSmis = []
    molsProd = []
    for smi in prodSet:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            print("cannot read", smi)
            continue
        prodSmis.append(smi)
        molsProd.append(mol)
    if not all(molsProd):
        print("1111", prodSmis)
        print("2222", molsProd)
        raise
    fullrx = []
    usedSbs = set()
    usedProd = set()
    for bi, b in enumerate(molsB):
        for xi, x in enumerate(molsX):
            trueProds = isSuzukiRx(b,x, molsProd, prodSmis)
            if trueProds:
                fullrx.append( (candidateB[bi], candidateX[xi], trueProds) )
                usedSbs.add(candidateB[bi])
                usedSbs.add(candidateX[xi])
                for p in trueProds:
                    usedProd.add(p)
    notUsedSbs = [s for s in sbsset if not s in usedSbs]
    notUsedProd = [s for s in prodSet if not s in usedProd]
    return fullrx, notUsedSbs, notUsedProd

def isSuzukiRx(b,x, expProds, prodSmis, verbose=False):
    try:
        prods = suzukiRx.RunReactants([b,x])
        if verbose and not prods:
            print("!!no prod:!", Chem.MolToSmiles(b), Chem.MolToSmiles(x))
        mainProd = [ p[0] for p in prods]
        #print("??????????", Chem.MolToSmiles(b), "||", Chem.MolToSmiles(x), ">>>", [Chem.MolToSmiles(m) for m in mainProd])
        _ = [ Chem.SanitizeMol(p) for p in mainProd]
        mainSmiles = list([s for s in set([Chem.MolToSmiles(p) for p in mainProd]) if s])
        mainProds = [Chem.MolFromSmiles(s) for s in mainSmiles]
        found = set()
        for main in mainProds:
            if not main:
                continue
            #print("MAIN", main, "Exp", expProds)
            for poz, expProd in enumerate(expProds):
                if expProd.HasSubstructMatch(main) and main.HasSubstructMatch(expProd):
                    found.add( prodSmis[poz])
            #print(main, "P", prods)
        return tuple(found)
    except:
        raise
        return False

def getPdCat(catSmi):
    pd2smi = [s for s in catSmi if 'Pd+' in s]
    #"naked Pd+ is typo - corect it into pd+2
    pd2smi = ['[Pd+2]' if s == '[Pd+]' else s for s in pd2smi]
    pd0smi = [s for s in catSmi if s.count('Pd') > s.count('Pd+')]
    nopdsmi = [s for s in catSmi if not 'Pd' in s]
    if DEBUG_PD:
        print("PD:DEBUG:0 input", catSmi, "===>>", pd0smi, pd2smi, nopdsmi)
    if not pd2smi:
        if len(pd0smi) == 1 and '[Pd]' in pd0smi:
            if DEBUG_PD:
                print("PD:DEBUG:1", nopdsmi.count('CC(=O)O)') )
            if nopdsmi.count('CC(=O)O') > 1:
                nopdsmi.remove('CC(=O)O')
                nopdsmi.remove('CC(=O)O')
                pd0smi = ['CC(=O)[O-]', '[Pd+2]', 'CC(=O)[O-]']
        if DEBUG_PD:
            print("PD:DEBUG:2 out", nopdsmi, pd0smi)
        return nopdsmi, pd0smi

    #ignore unknown Pd is there is more defined Pd species
    pd0smi = [ s for s in pd0smi if s != '[Pd]']

    anionic = [x for x in nopdsmi if '-]' in x]
    aromaticCanion = [s for s in anionic if '[c-]' in s]
    nanion = [s for s in anionic if '[N-]' in s]


    if True:
        ## here below is series of special cases for various ionic palladium speciels which consists of more than one smiles
        if len(pd2smi) == 1 and pd2smi[0] == '[Pd+2]':
            if not anionic:
                if pd0smi:
                    #raise NotImplementedError
                    print("CATPD0", pd0smi)
                return nopdsmi, pd2smi
            ##special case for Pd acetate
            acetateNum = nopdsmi.count('CC(=O)[O-]')
            acetateCF3num = nopdsmi.count('O=C([O-])C(F)(F)F')
            ohNum = nopdsmi.count('[OH-]')
            if acetateNum > 1:
                nopdsmi.remove('CC(=O)[O-]')
                nopdsmi.remove('CC(=O)[O-]')
                if pd0smi:
                    #raise NotImplementedError
                    print("SOME PdOac2", pd2smi, pd0smi, nopdsmi, catSmi )
                return nopdsmi, ['CC(=O)[O-]', '[Pd+2]', 'CC(=O)[O-]']
            if acetateCF3num > 1:
                nopdsmi.remove('O=C([O-])C(F)(F)F')
                nopdsmi.remove('O=C([O-])C(F)(F)F')
                if pd0smi:
                    print("SOME PdOAcCF3", pd2smi, pd0smi, nopdsmi)
                return nopdsmi, ['O=C([O-])C(F)(F)F', '[Pd+2]', 'O=C([O-])C(F)(F)F']
            if len(aromaticCanion) == 2:
                canion1 = aromaticCanion[0]
                canion2 = aromaticCanion[1]
                nopdsmi.remove(canion1)
                nopdsmi.remove(canion2)
                if pd0smi:
                    print("SOME pdcarbene", pd2smi, pd0smi, nopdsmi)
                return nopdsmi, [canion1, '[Pd+2]', canion2]
            clNum = nopdsmi.count('[Cl-]')

            if clNum >= 2 and len(anionic) >= 2 and len(set(anionic)) ==1:
                nopdsmi.remove('[Cl-]')
                nopdsmi.remove('[Cl-]')
                pd0smi = [s for s in pd0smi if s != 'Cl[Pd]Cl']
                if pd0smi:
                    print("SOME PdCl2", pd0smi, pd2smi, nopdsmi)
                    #raise NotImplementedError
                return nopdsmi, ['[Cl-]', '[Pd+2]', '[Cl-]']
            otherAnions = set([a for a in anionic if a != '[Cl-]'])
            allowedAnions = {'O=C([O-])[O-]', }
            if clNum >=2 and otherAnions.issubset(allowedAnions):
                nopdsmi.remove('[Cl-]')
                nopdsmi.remove('[Cl-]')
                return nopdsmi, ['[Cl-]', '[Pd+2]', '[Cl-]']

            if len(aromaticCanion) == 2:
                aroC1 = aromaticCanion[0]
                aroC2 = aromaticCanion[1]
                nopdsmi.remove(aroC1)
                nopdsmi.remove(aroC2)
                return nopdsmi, [aroC1, '[Pd+2]', aroC2]

            if len(nanion) == 2:
                counteranion1 = nanion[0]
                counteranion2 = nanion[1]
                nopdsmi.remove(counteranion1)
                nopdsmi.remove(counteranion2)
                return nopdsmi, [counteranion1, '[Pd+2]', counteranion2]

            if ohNum == 2 and len(anionic) == 2:
                nopdsmi.remove('[OH-]')
                nopdsmi.remove('[OH-]')
                if pd0smi:
                    print("SOME Pd(OH)2", pd0smi, pd2smi, nopdsmi)
                    #raise NotImplementedError
                return nopdsmi, ['[OH-]', '[Pd+2]', '[OH-]']
            #raise NotImplementedError

            if len(set(anionic)) == 1:
                if len(anionic) == 1:
                    counterion = anionic[0]
                    nopdsmi.remove(counterion)
                    countercharge = counterion.count('-')
                    if countercharge == 1:
                        return nopdsmi, [counterion, '[Pd+2]', counterion]
                    elif countercharge == 2:
                        return nopdsmi, [counterion, '[Pd+2]']
                    else:
                        raise NotImplementedError
                else:
                    counterion = anionic[0]
                    nopdsmi.remove(counterion)
                    countercharge = counterion.count('-')
                    if countercharge  == 1:
                        counterion2 = anionic[1]
                        nopdsmi.remove(counterion2)
                        return nopdsmi, [counterion, '[Pd+2]', counterion2]
                    elif countercharge == 2:
                        return nopdsmi, [counterion, '[Pd+2]']
                    else:
                        raise NotImplementedError

        ##Pd4+
        if len(pd2smi) == 1 and pd2smi[0] == 'Cl[Pd+2](Cl)(Cl)Cl':
            na = nopdsmi.count('[Na+]')
            nopdsmi.remove('[Na+]')
            nopdsmi.remove('[Na+]')
            return nopdsmi, [ '[Na+]', '[Na+]', 'Cl[Pd+2](Cl)(Cl)Cl']
        
        if len(pd2smi) == 1 and  'Cl[Pd+]' in pd2smi:
            if len(aromaticCanion) == 1:
                aroCanion = aromaticCanion[0]
                nopdsmi.remove(aroCanion)
                return nopdsmi, [aroCanion, 'Cl[Pd+]']
            if len(anionic) == 1:
                counterion = anionic[0]
                nopdsmi.remove(counterion)
                return nopdsmi, [ counterion, 'Cl[Pd+]']
        if len(set(pd2smi)) == 1 and 'Cl[Pd+]' in pd2smi:
            if len(aromaticCanion) == len(pd2smi):
                aro1 = aromaticCanion[0]
                aro2 = aromaticCanion[1]
                nopdsmi.remove(aro1)
                nopdsmi.remove(aro2)
                return nopdsmi, [(aro1, 'Cl[Pd+]'), (aro2, 'Cl[Pd+]')]

        if len(pd2smi) == 1 and 'Br[Pd+]':
            if  pd2smi and len(nanion) == 1:
                counteranion = nanion[0]
                nopdsmi.remove(counteranion)
                return nopdsmi, ['Br[Pd+]', counteranion]

        if len(pd2smi) == 1 and len(set(anionic)) == 1:
            print("ANI!!!",pd2smi, nopdsmi,"|||", catSmi)
            return nopdsmi, pd2smi
        #elif len(pd2smi) == 1
    print("!!!!ANI!", catSmi, pd2smi, pd0smi)
    raise
    return nopdsmi, -1

def isInSelectedRx(smiles, selectedRx):
    pass

def getYield(yields, oneValue=True, removeWrong=True):
    yld = [x for x in yields if x]
    yld = [yieldToFloat(x) for x in yld]
    if removeWrong:
        yld = [y for y in yld if y>0 and y<=100]
    if oneValue and yld:
        yld = statistics.mean(yld)
    return yld

def yieldToFloat(yld):
    yld = yld.replace('%', '').replace('<', '').replace('~', '').replace('>','').replace('=', '')
    if ' to ' in yld:
        yld = yld.split(' to ')
        if len(yld) != 2:
            print("!!!", yld)
        yld = 0.5 *(abs(float(yld[0])) + abs(float(yld[1])))

    elif '±' in yld:
        print("!!!", yld)
        yld = yld.split('±')[0]
    return float(yld)

def printClear(dct, elems=('Pd', 'ligands', 'solvents', 'bases')):
    sbs = '.'.join([dct['rx'][0][0],dct['rx'][0][1]])
    cond = []
    for elemType in elems:
        if elemType in dct and dct[elemType]:
            for elm in dct[elemType]:
                if isinstance(elm,str):
                    cond.append(elm)
                else:
                    cond.append( '.'.join(elm))
    cond = '.'.join(cond)
    prod = '.'.join(dct['rx'][0][2])
    return sbs+'>'+cond+'>'+prod+' '+str(dct['yield'])


def saveToFiles(rxlist, prefix, singleProduct=True, debug=False):
    frawhomo = open(prefix+'raw_homo.txt', 'w')
    frawhet = open(prefix+'raw_hete.txt', 'w')
    fparsedhomo = open(prefix+'parsed_homo.txt', 'w')
    fparsedhet = open(prefix+'parsed_het.txt', 'w')
    fclearhomo = open(prefix+'clear_homo.txt', 'w')
    fclearhet =  open(prefix+'clear_het.txt', 'w')
    #fdebug = open(prefix+'debug.txt', 'w')
    if debug:
        fother = open(prefix+'notdetectedconditions.txt', 'w')
        otherIgnore = {x.split()[0] for x in open('ignorable.other')}
        fsbs = open(prefix+'notdetectedsubstrate.txt', 'w')
        fprod = open(prefix+'notdetectedproduct.txt', 'w')
    for rx in rxlist:
        if singleProduct:
            if len(rx['parsed']['rx']) > 1:
                print("RX111", rx['parsed']['rx'])
                continue
            if len(rx['parsed']['rx'][0][2]) > 1:
                print("RX@@@2", rx['parsed']['rx'][0][2])
                continue
        sbs1 = Chem.MolFromSmiles( rx['parsed']['rx'][0][0])
        sbs2 = Chem.MolFromSmiles( rx['parsed']['rx'][0][1])
        if sbs1.HasSubstructMatch(het1) or sbs2.HasSubstructMatch(het1) or sbs1.HasSubstructMatch(het2) or sbs2.HasSubstructMatch(het2) or \
            sbs1.HasSubstructMatch(het3) or sbs2.HasSubstructMatch(het3) or sbs1.HasSubstructMatch(het4) or sbs2.HasSubstructMatch(het4):
            fraw = frawhet
            fclear = fclearhet
            fparsed = fparsedhet
        else:
            fraw = frawhomo
            fclear = fclearhomo
            fparsed = fparsedhomo
        print(rx['raw'][0].split()[0], rx['parsed']['yield'], file=fraw)
        print(rx['parsed'], file=fparsed)
        #print(rx['parsed'], file=fclear)
        print(printClear(rx['parsed']), file=fclear)
        if debug == True:
            if 'other' in rx['parsed']:
                trueOther = [x for x in rx['parsed']['other'] if x not in otherIgnore]
                showIt = [  ['[NH4+]', '[Cl-]'], ['CCCC[N+](CCCC)(CCCC)CCCC', '[Br-]'], ['[Na+]', '[Cl-]'], ['[Cl-]', '[Na+]'], ['[Br-]', 'CCCC[N+](CCCC)(CCCC)CCCC'],['[Cl-]', '[NH4+]'], ]
                if trueOther not in showIt:
                    print(trueOther, rx['parsed']['other'], rx['parsed'], rx['raw'][0].split()[0], sep='\t', file=fother)
            if 'notUsedSbs' in rx['parsed']:
                print(rx['parsed']['notUsedSbs'], rx['parsed'], rx['raw'][0].split()[0], sep='\t', file=fsbs)
            if 'notUsedProd' in rx['parsed']:
                print(rx['parsed']['notUsedProd'], rx['parsed'], rx['raw'][0].split()[0], sep='\t', file=fprod)
    frawhomo.close()
    frawhet.close()
    fclearhomo.close()
    fclearhet.close()
    fparsedhomo.close()
    fparsedhet.close()
    if debug:
        fother.close()
        fsbs.close()
        fprod.close()


if __name__ == "__main__":
    uspto_path = '/home/xiaoruiwang/data/ubuntu_work_beta/multi_step_work/handle_multi_step_dataset/USPTO_org/'    # USPTO rsmi path
    files = ['1976_Sep2016_USPTOgrants_smiles.rsmi','2001_Sep2016_USPTOapplications_smiles.rsmi']
    #files = ['testowe', ]
    #solventDict = parseCondFile('solventscanon.smi')
    #baseDict = parseCondFile('basescanon.smi')
    replaceObj = ReplaceCmd('uspto_replace_list.csv', 'solventscanon.smi', 'basescanon.smi', 'liganscanon.smi')
    outputPrefix ='../../dataset/source_dataset/USPTO_suzuki_final/suzuki_from_arom_USPTO_'
    if not os.path.exists(os.path.dirname(outputPrefix)):
        os.makedirs(os.path.dirname(outputPrefix))
    allRx = []
    seenSmiles = set()
    for f in files:
        f = os.path.join(uspto_path, f)
        data, seenSmiles = parseRxFile(f, seenSmiles, replace=replaceObj)
        allRx.extend(data)
        print(f, len(data))
    saveToFiles(allRx, outputPrefix, singleProduct=True)