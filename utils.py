import numpy as np
import pandas as pd, tabulate
from tabulate import tabulate
import statistics
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import domain
import re
from typing import List

def initializeControls(controls : List['domain.Control']) -> None:
    dfcontrol = pd.read_excel('controls.xlsx', sheet_name='nist')
    for row in dfcontrol.itertuples():
        control = domain.Control(row.controlId)
        control.baseline = row.baseline
        control.cat = row.category
        control.title = row.controlName
        controls.append(control)

def initializeTactics(tactics : List['domain.Tactic']) -> None:
    dfTactic = pd.read_excel('ttps.xlsx', sheet_name='tactic')
    for row in dfTactic.itertuples():
        tactic = domain.Tactic(row.tacticId)
        tactic.title = row.tacticName
        tactics.append(tactic)

def initializeTechniques(techniques : List['domain.Technique']) -> None:
    dfTechnique = pd.read_excel('ttps.xlsx', sheet_name='technique')
    for row in dfTechnique.itertuples():
        ifAny = [x for x in techniques if x.name == row.techniqueId]
        if len(ifAny) == 0:
            technique = domain.Technique(row.techniqueId)
            technique.title = row.techniqueName
            techniques.append(technique)

def initializeTacticTechniqueMapping(tactics : List['domain.Tactic'], techniques : List['domain.Technique']) -> None:
    dfTechnique = pd.read_excel('ttps.xlsx', sheet_name='technique')
    for row in dfTechnique.itertuples():
        technique = [x for x in techniques if x.name == row.techniqueId][0]
        tactic = [x for x in tactics if x.name == row.tactics][0]
        if tactic not in technique.tactics: technique.tactics.append(tactic)
        if technique not in tactic.techniques: tactic.techniques.append(technique)

def initializeControlTechniqueMapping(controls : List['domain.Control'], techniques : List['domain.Technique']) -> None:
    dfMapping = pd.read_excel('ttp-control.xlsx', sheet_name='map-nist')
    for row in dfMapping.itertuples():
        ifAny = [x for x in controls if x.name == row.controlId]
        if len(ifAny) > 0:
            control = ifAny[0]
            technique = [x for x in techniques if x.name == row.techniqueId][0]
            if technique not in control.ttps: control.ttps.append(technique)
            if control not in technique.controls: 
                technique.controls.append(control)
                for tactic in technique.tactics:
                    if control not in tactic.controls: tactic.controls.append(control)
    for technique in techniques:
        technique.scCount = len(technique.controls)

def initializeCocTTPs(groups, softwares, cocTTPs):
    allTechniques = []

    for g in groups:
        for te in g.ttps:
            allTechniques.append(te)

    allTechniques = list( set(allTechniques) )

    for s in softwares:
        for te in s.ttps:
            allTechniques.append(te)

    allTechniques = list( set(allTechniques) )
    allTechniques.sort(key=lambda t : t.name)

    cocTTPs.extend([g.ttps for g in groups])
    cocTTPs.extend([s.ttps for s in softwares])
    
    return

def calculateTEC(controls : List['domain.Control']) -> None:
    for control in controls:
        control.tec = len(control.ttps)
    return

def calculateTAC(controls : List['domain.Control'], tactics : List['domain.Tactic']) -> None:
    controls = [x for x in controls if x.tec > 0]
    tactics.sort(key = lambda t : t.name)
    tacticsNameList = [x.name for x in tactics]
    tacticsMemberCountList = [len(x.techniques) for x in tactics]
    
    for control in controls:
        for index in range(0, len(tacticsNameList)):
            count = 0
            for ttp in control.ttps:
                if tacticsNameList[index] in [x.name for x in ttp.tactics]:
                    count += 1
            control.tac.append( count / tacticsMemberCountList[index] )
        control.mtac = statistics.mean(control.tac)

def calculateImportance(controls : List['domain.Control'], techniques : List['domain.Technique']) -> None:
    controls = [x for x in controls if x.tec > 0]
    techniques = [x for x in techniques if x.scCount > 0]
    
    for ttp in techniques:    
        for control in ttp.controls:
            control.ti += 1/ttp.scCount

    for control in controls:
        control.ri = control.ti/control.tec

def find_citation(text : str) -> str:
    tokens = text.split('(Citation: ')
    tokens = tokens[-1].strip().split(' ')
    outputText = ''
    for index in range(0, len(tokens)):
        outputText = outputText + ' ' + tokens[index]
    return outputText[0:-1].strip()

def findYear(text : str) -> str:
    regexPattern = r'''[0-9][0-9][0-9][0-9]'''
    match = re.search(regexPattern, text)
    return text[match.span()[0] : match.span()[1]]


def initializeProcedures(procedures : List['domain.Procedure']) -> None:
    dfProcedures = pd.read_excel('technique.xlsx', sheet_name='procedure')
    dfProcedures = dfProcedures[['sourceId', 'targetId', 'citation']]
    dfProcedures['targetId'] = dfProcedures['targetId'].apply(lambda row : row[0:5])
    dfProcedures['citation'] = dfProcedures['citation'].apply(find_citation)
    dfPRef = pd.read_excel('technique.xlsx', sheet_name='citations')
    dfPRef['reference'] = dfPRef['reference'].apply(findYear)
    dfPRef = dfPRef.drop_duplicates()
    dfm = pd.merge(dfProcedures, dfPRef, how='left', left_on=['citation'], right_on=['citation'])
    dfm.drop_duplicates()
    
    for row in dfm.itertuples():
        procedure = domain.Procedure(row.sourceId + ':' + row.targetId + ':' + '-'.join(str(row.citation).split(' ')))
        procedure.ttp = row.targetId
        procedure.year = row.reference
        procedure.name = row.sourceId + ':' + row.targetId
        if 'G' in procedure.id: 
            procedure.type = 'group'
        else: 
            procedure.type = 'software'
        procedures.append(procedure)

def initializeGroups(groups : List['domain.Group'], techniques : List['domain.Technique']) -> None:
    dfGroups = pd.read_excel('groups.xlsx', sheet_name='ttps')
    dfGroups = dfGroups[['sourceId', 'targetId']]
    dfGroups['targetId'] = dfGroups['targetId'].apply(lambda v : v[0:5])
    dfGroups = dfGroups.drop_duplicates()
    dfg = dfGroups.groupby(['sourceId'])
    
    for name, group in dfg:
        g = domain.Group(name)
        
        for row in group.itertuples():
            ttp = [x for x in techniques if x.name == row.targetId][0]
            if ttp not in g.ttps: 
                g.ttps.append(ttp)
                for control in ttp.controls:
                    if control not in g.controls:
                        g.controls.append(control)
        groups.append(g)

def initializeSoftwares(softwares : List['domain.Software'], techniques : List['domain.Technique']) -> None:
    dfSoftwares = pd.read_excel('software.xlsx', sheet_name='ttps')
    dfSoftwares = dfSoftwares[['sourceId', 'targetId']]
    dfSoftwares['targetId'] = dfSoftwares['targetId'].apply(lambda v : v[0:5])
    dfSoftwares = dfSoftwares.drop_duplicates()
    dfg = dfSoftwares.groupby(['sourceId'])
    
    for name, group in dfg:
        software = domain.Software(name)
        
        for row in group.itertuples():
            ttp = [x for x in techniques if x.name == row.targetId][0]
            if ttp not in software.ttps: 
                software.ttps.append(ttp)
                for control in ttp.controls:
                    if control not in software.controls:
                        software.controls.append(control)
        softwares.append(software)

def calculateProb(controls : List['domain.Control'], techniques : List['domain.Technique'], procedures : List['domain.Procedure']) -> None:
    for p in procedures:
        ttp = [x for x in techniques if x.name == p.ttp][0]
        ttp.years.append(p.year)
        ttp.occurrence += 1
    
    for ttp in techniques:
        ttp.prob = ttp.occurrence / len(procedures)
    
    for control in controls:
        count = 0
        for ttp in control.ttps:
            count += ttp.occurrence
        control.prob = count / len(procedures)
    
    return

def evaluateControlRedundancy(controls : List['domain.Control']):
    mcontrols = [x for x in controls if len(x.ttps) > 0]

    for control in mcontrols:
        redundancyCount = [len(x.controls) - 1 for x in control.ttps]
        control.red = sum(redundancyCount)/len(redundancyCount)
    return

def computeTechniqueLikelihood(techniques, cocTTPs):
    for te in techniques:
        count = 0
        for adv in cocTTPs:
            if te in adv:
                count += 1
        te.lk = count/len(cocTTPs)
    return

def buildDataSchema(controls : List['domain.Control'], tactics : List['domain.Tactic'], techniques : List['domain.Technique'], procedures : List['domain.Procedure'], groups : List['domain.Group'], softwares : List['domain.Software'], cocTTPs : List[List['domain.Technique']]) -> None:
    initializeControls(controls)
    initializeTactics(tactics)
    initializeTechniques(techniques)
    initializeTacticTechniqueMapping(tactics, techniques)
    initializeControlTechniqueMapping(controls, techniques)
    calculateTEC(controls)
    calculateTAC(controls, tactics)
    calculateImportance(controls, techniques)
    initializeProcedures(procedures)
    initializeGroups(groups, techniques)
    initializeSoftwares(softwares, techniques)
    calculateProb(controls, techniques, procedures)
    evaluateControlRedundancy(controls)
    initializeCocTTPs(groups, softwares, cocTTPs)
    computeTechniqueSupport(techniques, cocTTPs)
    computeTechniqueSeverity(techniques, tactics, cocTTPs)
    return


def normalizeList(list: List['float']) -> List[float]:
    minimum = min(list)
    maximum = max(list)
    normalizeList = []
    for item in list:
        normalizeList.append((item - minimum) / (maximum - minimum))
    return normalizeList


def minSetCover(universe : set['domain.Technique'], subsets: set['domain.Control'], flag):
    if flag == 'all': 
        elements = set(te for c in subsets for te in c.ttps)
    
    if flag.startswith('TA'): 
        elements = set(te for c in subsets for te in c.ttps if te.tactics[0].name == flag)
    
    if elements != universe: 
        return None
    
    covered = set()
    cover = []
    
    while covered != elements:
        subset = max(subsets, key = lambda c : len(set(c.ttps) - set(covered)))
        cover.append(subset)
        covered |= set(subset.ttps)
        
    return cover

def computeTechniqueSupport(techniques, cocTTPs):
    for te in techniques:
        count = 0
        for adv in cocTTPs:
            if te in adv:
                count += 1
        te.support = count / len(cocTTPs)
    return

def computeTechniqueSeverity(techniques, tactics, cocTTPs):
    tactics.sort(key = lambda t : t.name)
    for te1 in techniques:
        timesTe1 = 0
        cooc = []
        
        for adv in cocTTPs:
            if te1 in adv: timesTe1 += 1
        
        for ta in tactics:
            timesTe1AndTa = 0
            for adv in cocTTPs:
                if (te1 in adv) and (set(ta.techniques).isdisjoint(set(adv))) == False:
                    timesTe1AndTa += 1
            if timesTe1 > 30: cooc.append(timesTe1AndTa/timesTe1)
            else: cooc.append(0)
        
        te1.severity = statistics.median(cooc)
    return

def printTopN(objList, N, attr, attrType):
    for idx in range(0, N):
        if attrType == 'item': print(f'{objList[idx].name}: {objList[idx].title} ==> {getattr(objList[idx], attr)}')
        if attrType == 'list': print(f'{objList[idx].name}: {objList[idx].title} ==> {getattr(objList[idx], attr)[0].name}: {getattr(objList[idx], attr)[0].title}')
    return